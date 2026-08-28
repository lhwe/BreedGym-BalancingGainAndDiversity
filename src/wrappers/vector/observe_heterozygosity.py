from gymnasium.vector import VectorWrapper
from gymnasium import spaces
import numpy as np

from src.utils.hetobs import individual_heterozygosity


class ObserveHeterozygosity(VectorWrapper):
    """
    Adds a per-individual heterozygosity to the observation.
    Turns obs into {"obs": raw_obs, "heterozygosity": het_per_individual}.
    Downstream wrappers (MaskSNPsHet, ObserveGenNumberHet) expect
    this dict shape.
    """

    def __init__(self, env):
        super().__init__(env)
        ind_per_gen = self.unwrapped.individual_per_gen

        self.single_observation_space = spaces.Dict({
            "obs": self.env.single_observation_space,
            "heterozygosity": spaces.Box(
                low=0.0, high=1.0, shape=(ind_per_gen, 1), dtype=np.float32
            )
        })
        self.observation_space = spaces.Dict({
            "obs": self.env.observation_space,
            "heterozygosity": spaces.Box(
                low=0.0, high=1.0, shape=(self.num_envs, ind_per_gen, 1), dtype=np.float32
            )
        })

    def set_attr(self, name, value, indices=None):
        return self.env.set_attr(name, value, indices)

    def _make_obs(self, obs):
        het = individual_heterozygosity(obs)[..., None]  # (num_envs, ind_per_gen, 1)
        return {"obs": obs, "heterozygosity": het}

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._make_obs(obs), info

    def step(self, actions):
        obs, rew, ter, tru, infos = self.env.step(actions)
        new_obs = self._make_obs(obs)

        if "final_observation" in infos:
            infos["final_observation"] = self._make_obs(infos["final_observation"])

        return new_obs, rew, ter, tru, infos