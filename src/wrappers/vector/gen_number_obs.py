from gymnasium.vector import VectorWrapper
from gymnasium import spaces
import numpy as np


class ObserveGenNumber(VectorWrapper):

    def __init__(self, env, enabled=True):
        """
        :param env: wrapped env.
        :param enabled: Whether to actually add the "gen_number" observation.
            - True: Add "gen_number" to the Dict
            - False: Only return the original obs
        """
        super().__init__(env)
        self.enabled = enabled
        if self.enabled:
            self.observation_space = spaces.Dict({
                "obs": self.env.observation_space,
                "gen_number": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(self.num_envs, 1),
                    dtype=np.float32
                )
            })

            self.single_observation_space = spaces.Dict({
                "obs": self.env.single_observation_space,
                "gen_number": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(1,),
                    dtype=np.float32
                )
            })
        else: # still return dict
            self.observation_space = spaces.Dict({
                "obs": self.env.observation_space,
            })
            self.single_observation_space = spaces.Dict({
                "obs": self.env.single_observation_space,
            })
        self.gen_number = None

    def set_attr(self, name, value, indices=None):
        return self.env.set_attr(name, value, indices)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        if self.enabled:
            self.gen_number = -np.ones((self.num_envs, 1), dtype=np.float32)
            return {"obs": obs, "gen_number": self.gen_number.copy()}, info
        else:
            return {"obs": obs}, info
        return {"obs": obs, "gen_number": self.gen_number.copy()}, info

    def step(self, actions):
        obs, rew, ter, tru, infos = self.env.step(actions)

        if self.enabled:
            dones = np.logical_or(ter, tru)
            increment = 2 / (self.unwrapped.num_generations - 1)
            self.gen_number = np.where(
                dones[:, np.newaxis],  # expand dims to match (num_envs, 1)
                -1.0, # if done==True, reset to -1
                self.gen_number + increment
            ).astype(np.float32)

            new_obs = {"obs": obs, "gen_number": self.gen_number.copy()}

            if "final_observation" in infos:
                # The gen_number at the end of an episode is usually 1.0
                term_gen = np.full((self.num_envs, 1), 1.0, dtype=np.float32)
                infos["final_observation"] = {
                    "obs": infos["final_observation"],
                    "gen_number": term_gen
                }

            return new_obs, rew, ter, tru, infos

        else:
            new_obs = {"obs": obs}

            if "final_observation" in infos:
                infos["final_observation"] = {
                    "obs": infos["final_observation"]
                }

            return new_obs, rew, ter, tru, infos