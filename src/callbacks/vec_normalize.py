import os
from stable_baselines3.common.callbacks import BaseCallback


class SaveBestVecNormalizeCallback(BaseCallback):
    """Callback for saving a VecNormalize wrapper for best model."""
    def __init__(self, save_dir: str, verbose: int = 1):
        super(SaveBestVecNormalizeCallback, self).__init__(verbose)
        self.save_path = os.path.join(save_dir, "vec_normalize.pkl")

    def _on_step(self) -> bool:
        vec_normalize_env = self.model.get_vec_normalize_env()
        if vec_normalize_env is not None:
            vec_normalize_env.save(self.save_path)
            if self.verbose > 0:
                print(f"Detected new best model! Saving VecNormalize stats to {self.save_path}")
        return True
