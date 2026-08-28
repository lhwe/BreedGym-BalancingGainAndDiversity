import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import wandb


class LoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")

        if infos is not None:
            absolute_GEBV = np.mean([info.get("absolute_GEBV", 0) for info in infos])
            gebv_std = np.mean([info.get("GEBV_std", 0) for info in infos])
            # delta_g = np.mean([info.get("delta_G", 0) for info in infos])
            # delta_std = np.mean([info.get("delta_std", 0) for info in infos])
            # delta_He = np.mean([info.get("delta_He", 0) for info in infos])
            expected_heterozygosity = np.mean([info.get("expected_heterozygosity", 0) for info in infos])
            # inbreeding_penalty = np.mean([info.get("inbreeding_penalty", 0) for info in infos])
            # ema_w_inbreed = np.mean([info.get("ema_w_inbreed", 0) for info in infos])
            # inbreed_markers = np.mean([info.get("inbreed_markers", 0) for info in infos])
            # delta_inbreed_markers = np.mean([info.get("delta_inbreed_markers", 0) for info in infos])
            # pedigree_penalty = np.mean([info.get("pedigree_penalty", 0) for info in infos])

            self.logger.record("custom_env/absolute_GEBV", absolute_GEBV)
            self.logger.record("custom_env/GEBV_std", gebv_std)
            # self.logger.record("custom_env/delta_G", delta_g)
            # self.logger.record("custom_env/delta_std", delta_std)
            # self.logger.record("custom_env/delta_He", delta_He)
            self.logger.record("custom_env/expected_heterozygosity", expected_heterozygosity)
            # self.logger.record("custom_env/inbreeding_penalty", inbreeding_penalty)
            # self.logger.record("custom_env/ema_w_inbreed", ema_w_inbreed)
            # self.logger.record("custom_env/inbreed_markers", inbreed_markers)
            # self.logger.record("custom_env/delta_inbreed_markers", delta_inbreed_markers)
            # self.logger.record("custom_env/pedigree_penalty", pedigree_penalty)

        return True