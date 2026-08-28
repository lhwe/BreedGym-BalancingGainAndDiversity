from stable_baselines3.common.vec_env.base_vec_env import VecEnv
import numpy as np
import torch


class AdaptVecEnv(VecEnv):
    
    def __init__(self, gym_vec_env):
        self.gym_vec_env = gym_vec_env
        self._actions = None
        self._seed = None
        
        super().__init__(
            self.gym_vec_env.num_envs,
            self.gym_vec_env.single_observation_space,
            self.gym_vec_env.single_action_space
        )
        
    def reset(self):
        obs, _ = self.gym_vec_env.reset(seed=self._seed)
        self._seed = None
        return self._adapt_obs(obs)

    def step_async(self, actions):
        self._actions = actions

    def step_wait(self):
        obs, rew, ter, tru, infos = self.gym_vec_env.step(self._actions)

        ter_np = np.asarray(ter)
        tru_np = np.asarray(tru)
        dones = np.logical_or(ter_np, tru_np)

        adapt_infos = []
        for i in range(self.num_envs):
            info_i = {}
            for k, v in infos.items():
                if not k.startswith('_') and k not in ["final_observation", "final_info"]:
                    try:
                        info_i[k] = v[i]
                    except (TypeError, IndexError):
                        info_i[k] = v

            if tru_np[i]:
                info_i["TimeLimit.truncated"] = True

            if dones[i]:
                if "final_observation" in infos and infos.get("_final_observation", np.zeros(self.num_envs))[i]:
                    info_i["terminal_observation"] = self._extract_single_obs(infos["final_observation"], i)
                else:
                    info_i["terminal_observation"] = self._extract_single_obs(obs, i)

            adapt_infos.append(info_i)

        new_obs = self._adapt_obs(obs)
        return new_obs, np.asarray(rew), dones, adapt_infos

    def _extract_single_obs(self, obs, index):
        if isinstance(obs, dict):
            return {k: self._extract_single_obs(v, index) for k, v in obs.items()}
        return np.asarray(obs[index])

    def _adapt_obs(self, obs):
        if isinstance(obs, dict):
            return {key: self._adapt_obs(value) for key, value in obs.items()}
        return np.asarray(obs)
        
        # if torch.cuda.is_available():
        #     if hasattr(obs, '__dlpack__'):
        #         return torch.from_dlpack(obs)
        #     return torch.as_tensor(np.asarray(obs), device='cuda')
        # else:
            # return np.asarray(obs)
        return np.asarray(obs)

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 10000)
        self._seed = seed
        return [seed + i for i in range(self.num_envs)]

    def close(self):
        return self.gym_vec_env.close()

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(f"Attempted to get missing private attribute '{name}'")
        return getattr(self.gym_vec_env, name)

    def get_attr(self, attr_name: str, indices=None):
        len_indices = len(self._get_indices(indices))
        return [getattr(self.gym_vec_env, attr_name)] * len_indices

    def set_attr(self, attr_name: str, value, indices=None):
        self.gym_vec_env.set_attr(attr_name, value)

    def env_method(self, method_name: str, *args, indices=None, **kwargs):
        return getattr(self.gym_vec_env, method_name)(*args, **kwargs)

    def env_is_wrapped(self, wrapper_class, indices=None):
        len_indices = len(self._get_indices(indices))
        return [isinstance(self.gym_vec_env, wrapper_class)] * len_indices

    def _get_indices(self, indices):
        if indices is None:
            return np.arange(self.num_envs)
        elif isinstance(indices, int):
            return [indices]
        else:
            return indices