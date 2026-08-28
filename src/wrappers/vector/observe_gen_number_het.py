from gymnasium.vector import VectorWrapper
from gymnasium import spaces
import numpy as np


class ObserveGenNumberHet(VectorWrapper):
    """
    Dict-aware version of ObserveGenNumber. Merges "gen_number" into the
    existing {"obs": ..., "heterozygosity": ...} dict.
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Dict({
            **self.env.observation_space.spaces,
            "gen_number": spaces.Box(
                low=-1.0, high=1.0, shape=(self.num_envs, 1), dtype=np.float32
            )
        })
        self.single_observation_space = spaces.Dict({
            **self.env.single_observation_space.spaces,
            "gen_number": spaces.Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float32
            )
        })
        self.gen_number = None

    def set_attr(self, name, value, indices=None):
        return self.env.set_attr(name, value, indices)

    def reset(self, *, seed=None, options=None):
        self.gen_number = -np.ones((self.num_envs, 1), dtype=np.float32)
        obs, info = self.env.reset(seed=seed, options=options)
        return {**obs, "gen_number": self.gen_number.copy()}, info

    def step(self, actions):
        obs, rew, ter, tru, infos = self.env.step(actions)

        dones = np.logical_or(ter, tru)
        increment = 2 / (self.unwrapped.num_generations - 1)
        self.gen_number = np.where(
            dones[:, np.newaxis],
            -1.0,
            self.gen_number + increment
        ).astype(np.float32)

        new_obs = {**obs, "gen_number": self.gen_number.copy()}

        if "final_observation" in infos:
            term_gen = np.full((self.num_envs, 1), 1.0, dtype=np.float32)
            infos["final_observation"] = {
                **infos["final_observation"],
                "gen_number": term_gen
            }

        return new_obs, rew, ter, tru, infos