from gymnasium import spaces
from gymnasium.vector import VectorObservationWrapper
import numpy as np


class MaskSNPsHet(VectorObservationWrapper):
    """
    Dict-aware version of MaskSNPs. ({"obs": ..., "heterozygosity": ...})
    as produced by ObserveHeterozygosity.
    """

    def __init__(self, env):
        super().__init__(env)

        mrks = self.unwrapped.simulator.GEBV_model.marker_effects
        self.mrk_std = mrks.std()

        low_val = float(min(0, mrks.min() / self.mrk_std))
        high_val = float(max(0, mrks.max() / self.mrk_std))

        masked_single = spaces.Box(
            low=low_val, high=high_val,
            shape=self.env.single_observation_space["obs"].shape,
            dtype=np.float32
        )
        masked_batched = spaces.Box(
            low=low_val, high=high_val,
            shape=self.env.observation_space["obs"].shape,
            dtype=np.float32
        )

        self.single_observation_space = spaces.Dict({
            "obs": masked_single,
            "heterozygosity": self.env.single_observation_space["heterozygosity"]
        })
        self.observation_space = spaces.Dict({
            "obs": masked_batched,
            "heterozygosity": self.env.observation_space["heterozygosity"]
        })

    def observations(self, observations):
        raw_obs = observations["obs"]
        masked_obs = raw_obs * self.unwrapped.simulator.GEBV_model.marker_effects[None, None, :]
        return {
            "obs": masked_obs.astype(np.float32),
            "heterozygosity": observations["heterozygosity"]
        }

    def set_attr(self, name, value, indices=None):
        return self.env.set_attr(name, value, indices)

    def step(self, actions):
        obs, rew, ter, tru, infos = self.env.step(actions)
        obs = self.observations(obs)

        if "final_observation" in infos:
            infos["final_observation"] = self.observations(infos["final_observation"])

        return obs, rew, ter, tru, infos