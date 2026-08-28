from __future__ import annotations

import numpy as np
from gymnasium.vector import VectorWrapper


class FullSibPenaltyRewardWrapper(VectorWrapper):
    """Reward genetic gain while penalizing full-sib matings.

    This is a lightweight pedigree-only baseline. It does not try to control
    marker diversity directly; it only discourages mating individuals with the
    same recorded parent pair.
    """

    def __init__(self, env, w_gain: float = 1.0, w_sib_penalty: float = 5.0):
        super().__init__(env)
        self.w_gain = float(w_gain)
        self.w_sib_penalty = float(w_sib_penalty)
        self._previous_mean = None
        self.current_pedigree = None

    def reset(self, **kwargs):
        obs, infos = self.env.reset(**kwargs)
        self._previous_mean = self._mean_gebv(infos)
        self.current_pedigree = self._empty_pedigree()
        return obs, infos

    def step(self, actions):
        sib_mating_counts = self._count_full_sib_matings(actions)
        obs, original_reward, terminated, truncated, infos = self.env.step(actions)

        current_mean = self._mean_gebv(infos)
        delta_g = current_mean - self._previous_mean
        pedigree_penalty = self.w_sib_penalty * sib_mating_counts
        reward = self.w_gain * delta_g - pedigree_penalty

        infos["delta_G"] = delta_g
        infos["sib_mating_counts"] = sib_mating_counts
        infos["sib_mating_penalty"] = pedigree_penalty
        infos["reward_component_gain"] = self.w_gain * delta_g

        self.current_pedigree = np.asarray(actions).copy()
        self._previous_mean = current_mean
        self._handle_autoreset(terminated, truncated)
        return obs, reward, terminated, truncated, infos

    def _count_full_sib_matings(self, actions):
        actions = np.asarray(actions)
        if actions.ndim != 3 or actions.shape[-1] != 2:
            raise ValueError(
                "Pedigree reward wrappers must receive parent-pair actions with "
                "shape [num_envs, n_crosses, 2]. Put the pedigree wrapper inside "
                "SelectionScores: VecBreedGym -> PedigreeWrapper -> SelectionScores."
            )
        actions = actions.astype(np.int32, copy=False)
        env_idx = np.arange(self.unwrapped.num_envs)[:, None]
        p1_idx = actions[..., 0]
        p2_idx = actions[..., 1]

        p1_ancestry = self.current_pedigree[env_idx, p1_idx]
        p2_ancestry = self.current_pedigree[env_idx, p2_idx]
        p1_sorted = np.sort(p1_ancestry, axis=-1)
        p2_sorted = np.sort(p2_ancestry, axis=-1)

        is_full_sib = (
            (p1_sorted[..., 0] == p2_sorted[..., 0])
            & (p1_sorted[..., 1] == p2_sorted[..., 1])
            & (p1_sorted[..., 0] != -1)
        )
        return np.sum(is_full_sib, axis=-1).astype(np.float32)

    def _handle_autoreset(self, terminated, truncated):
        is_done = np.any(terminated) or np.any(truncated)
        if is_done and getattr(self.env.unwrapped, "autoreset", False):
            reset_infos = getattr(self.env.unwrapped, "reset_infos", None)
            if reset_infos is not None:
                self._previous_mean = self._mean_gebv(reset_infos)
                self.current_pedigree = self._empty_pedigree()

    def _empty_pedigree(self):
        return np.full(
            (self.unwrapped.num_envs, self.unwrapped.individual_per_gen, 2),
            -1,
            dtype=np.int32,
        )

    @staticmethod
    def _mean_gebv(infos):
        gebv = infos.get("GEBV")
        if gebv is None:
            raise KeyError("Cannot find 'GEBV' in env info.")
        if hasattr(gebv, "to_numpy"):
            return np.asarray(np.mean(gebv.to_numpy()))
        return np.asarray(np.mean(gebv, axis=(1, 2)))

    def set_attr(self, name, value, indices=None):
        return self.env.set_attr(name, value, indices)


class HybridHeFullSibRewardWrapper(VectorWrapper):
    """Adaptive genomic-diversity reward plus full-sib pedigree penalty.

    This is the main variant worth trying. The HE weight is scaled to the
    observed magnitudes of genetic gain and HE change, but unlike the older
    wrapper it does not decay to zero at the final generation.
    """

    def __init__(
        self,
        env,
        w_gain: float = 1.0,
        he_prop: float = 0.2,
        w_sib_penalty: float = 5.0,
        ema_alpha: float = 0.05,
        min_he_weight: float = 10.0,
        max_he_weight: float = 1000.0,
    ):
        super().__init__(env)
        self.w_gain = float(w_gain)
        self.he_prop = float(he_prop)
        self.w_sib_penalty = float(w_sib_penalty)
        self.ema_alpha = float(ema_alpha)
        self.min_he_weight = float(min_he_weight)
        self.max_he_weight = float(max_he_weight)

        self.ema_delta_g = 5.0
        self.ema_delta_he = 0.05
        self._previous_mean = None
        self._previous_he = None
        self.current_pedigree = None

    def reset(self, **kwargs):
        obs, infos = self.env.reset(**kwargs)
        self._previous_mean = self._mean_gebv(infos)
        self._previous_he = self._expected_heterozygosity(self.env.unwrapped.populations)
        self.current_pedigree = self._empty_pedigree()
        return obs, infos

    def step(self, actions):
        sib_mating_counts = self._count_full_sib_matings(actions)
        obs, original_reward, terminated, truncated, infos = self.env.step(actions)

        current_mean = self._mean_gebv(infos)
        current_he = self._expected_heterozygosity(self.env.unwrapped.populations)
        delta_g = current_mean - self._previous_mean
        delta_he = current_he - self._previous_he

        self.ema_delta_g = (
            (1.0 - self.ema_alpha) * self.ema_delta_g
            + self.ema_alpha * float(np.mean(np.abs(delta_g)))
        )
        self.ema_delta_he = (
            (1.0 - self.ema_alpha) * self.ema_delta_he
            + self.ema_alpha * float(np.mean(np.abs(delta_he)))
        )

        he_weight = self.he_prop * (max(self.ema_delta_g, 5.0) / (self.ema_delta_he + 1e-6))
        he_weight = float(np.clip(he_weight, self.min_he_weight, self.max_he_weight))

        gain_component = self.w_gain * delta_g
        he_component = he_weight * delta_he
        sib_penalty = self.w_sib_penalty * sib_mating_counts
        reward = gain_component + he_component - sib_penalty

        infos["delta_G"] = delta_g
        infos["absolute_He"] = current_he
        infos["delta_He"] = delta_he
        infos["adaptive_he_weight"] = np.full(self.unwrapped.num_envs, he_weight, dtype=np.float32)
        infos["sib_mating_counts"] = sib_mating_counts
        infos["sib_mating_penalty"] = sib_penalty
        infos["reward_component_gain"] = gain_component
        infos["reward_component_he"] = he_component

        self.current_pedigree = np.asarray(actions).copy()
        self._previous_mean = current_mean
        self._previous_he = current_he
        self._handle_autoreset(terminated, truncated)
        return obs, reward, terminated, truncated, infos

    def _count_full_sib_matings(self, actions):
        actions = np.asarray(actions)
        if actions.ndim != 3 or actions.shape[-1] != 2:
            raise ValueError(
                "Pedigree reward wrappers must receive parent-pair actions with "
                "shape [num_envs, n_crosses, 2]. Put the pedigree wrapper inside "
                "SelectionScores: VecBreedGym -> PedigreeWrapper -> SelectionScores."
            )
        actions = actions.astype(np.int32, copy=False)
        env_idx = np.arange(self.unwrapped.num_envs)[:, None]
        p1_idx = actions[..., 0]
        p2_idx = actions[..., 1]

        p1_ancestry = self.current_pedigree[env_idx, p1_idx]
        p2_ancestry = self.current_pedigree[env_idx, p2_idx]
        p1_sorted = np.sort(p1_ancestry, axis=-1)
        p2_sorted = np.sort(p2_ancestry, axis=-1)

        is_full_sib = (
            (p1_sorted[..., 0] == p2_sorted[..., 0])
            & (p1_sorted[..., 1] == p2_sorted[..., 1])
            & (p1_sorted[..., 0] != -1)
        )
        return np.sum(is_full_sib, axis=-1).astype(np.float32)

    def _handle_autoreset(self, terminated, truncated):
        is_done = np.any(terminated) or np.any(truncated)
        if is_done and getattr(self.env.unwrapped, "autoreset", False):
            reset_infos = getattr(self.env.unwrapped, "reset_infos", None)
            if reset_infos is not None:
                self._previous_mean = self._mean_gebv(reset_infos)
                self._previous_he = self._expected_heterozygosity(
                    self.env.unwrapped.populations
                )
                self.current_pedigree = self._empty_pedigree()

    def _empty_pedigree(self):
        return np.full(
            (self.unwrapped.num_envs, self.unwrapped.individual_per_gen, 2),
            -1,
            dtype=np.int32,
        )

    @staticmethod
    def _mean_gebv(infos):
        gebv = infos.get("GEBV")
        if gebv is None:
            raise KeyError("Cannot find 'GEBV' in env info.")
        if hasattr(gebv, "to_numpy"):
            return np.asarray(np.mean(gebv.to_numpy()))
        return np.asarray(np.mean(gebv, axis=(1, 2)))

    @staticmethod
    def _expected_heterozygosity(populations):
        pop_np = np.asarray(populations)
        if pop_np.ndim == 4:
            p = np.mean(pop_np, axis=(1, 3))
            return np.mean(2.0 * p * (1.0 - p), axis=-1)
        if pop_np.ndim == 3:
            p = np.mean(pop_np, axis=(0, 2))
            return np.asarray(np.mean(2.0 * p * (1.0 - p)))
        raise ValueError(f"Unexpected populations shape: {pop_np.shape}")

    def set_attr(self, name, value, indices=None):
        return self.env.set_attr(name, value, indices)


# Backward-compatible names for experiment scripts.
BanFullSibsRewardWrapper = FullSibPenaltyRewardWrapper
UltimateBreedingRewardWrapper = HybridHeFullSibRewardWrapper
PedigreeInbreedingRewardWrapper = HybridHeFullSibRewardWrapper
