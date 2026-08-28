import numpy as np


def expected_heterozygosity(obs):
    p = np.mean(obs, axis=(1, 3))  # obs shape: (num_envs, individuals, markers, 2)
    he = np.mean(2 * p * (1 - p), axis=1) # shape: (num_envs,)
    return he

def mean_gebv(gebv):
    return np.mean(gebv, axis=(1, 2))

def gebv_std(gebv):
    return np.std(gebv, axis=(1, 2))


class Rewards:
    @staticmethod
    def mean_gebv_reward(gebv_array: np.ndarray, obs_array: np.ndarray,
                        w_gain: float, w_std: float, w_inbreed: float) -> tuple:
        """
        Args:
            gebv_array: shape (num_envs, individuals, traits)
            obs_array: shape (num_envs, individuals, markers, 2)
        """
        gebv = mean_gebv(gebv_array)
        std = gebv_std(gebv_array)
        he = expected_heterozygosity(obs_array)

        rews = (w_gain * gebv) + (w_std * std) + (w_inbreed * he)

        metrics = {
            "mean_gebv": gebv,
            "gebv_std": std,
            "he": he
        }
        return rews, metrics