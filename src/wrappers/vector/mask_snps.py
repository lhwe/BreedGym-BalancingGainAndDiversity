from gymnasium import spaces
from gymnasium.vector import VectorObservationWrapper
import numpy as np

class MaskSNPs(VectorObservationWrapper):
    
    def __init__(self, env):
        super().__init__(env)
        
        mrks = self.unwrapped.simulator.GEBV_model.marker_effects
        self.mrk_std = mrks.std()

        low_val = float(min(0, mrks.min() / self.mrk_std))
        high_val = float(max(0, mrks.max() / self.mrk_std))
        self.single_observation_space = spaces.Box(
            low=low_val,
            high=high_val,
            shape=self.env.single_observation_space.shape, 
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=low_val,
            high=high_val,
            shape=self.env.observation_space.shape,
            dtype=np.float32
        )

    def observations(self, observations):
        masked_obs = observations * self.unwrapped.simulator.GEBV_model.marker_effects[None, None, :]

        # rec_vec = np.broadcast_to(
        #     self.unwrapped.simulator.recombination_vec[None, None, :],
        #     observations.shape[:-1]
        # )
        # return np.concatenate(
        #     (masked_obs, rec_vec[..., None]),
        #     axis=-1
        # )
        
        return masked_obs.astype(np.float32)

    def set_attr(self, name, value, indices=None):
        return self.env.set_attr(name, value, indices)

    def step(self, actions):
        obs, rew, ter, tru, infos = self.env.step(actions)
        obs = self.observations(obs)

        # Transform the terminal observation if it exists
        if "final_observation" in infos:
            infos["final_observation"] = self.observations(infos["final_observation"])

        return obs, rew, ter, tru, infos