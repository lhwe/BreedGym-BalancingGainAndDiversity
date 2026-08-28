import os
import torch
import wandb

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.vec_env import VecNormalize
from wandb.integration.sb3 import WandbCallback

from breedgym.utils.paths import DATA_PATH, MODEL_PATH, LOG_PATH
from src.callbacks.vec_normalize import SaveBestVecNormalizeCallback
from src.callbacks.adapt_gen_number import AdaptGenNumber
from src.networks.selection_model import SelectionAC, CNNFeaturesExtractor
from src.utils.config import load_config, namespace_to_dict
from src.utils.build_env import build_env
from src.utils.args import parse_args


def main(config):
    run_name = f"{config.unique_name}_seed_{config.seed}"

    run_log_path = config.log_path / f"{config.unique_name}_seed_{config.seed}"
    run_model_path = config.model_path / f"{config.unique_name}_seed_{config.seed}"

    os.makedirs(run_model_path, exist_ok=True)
    os.makedirs(run_log_path, exist_ok=True)

    if not config.disable_wandb:
        wandb.init(
            project=config.project,
            entity=config.entity,
            config=namespace_to_dict(config),
            notes=run_name,
            sync_tensorboard=True,
        )

    conv1_kwargs = {
        "out_channels": int(config.out_channels1),
        "kernel_size": int(config.kernel_size1),
        "stride": int(config.stride1),
    }
    conv2_kwargs = {
        "out_channels": int(config.out_channels2),
        "kernel_size": int(config.kernel_size2),
        "stride": int(config.stride2),
    }
    use_gen_obs = getattr(config, "use_observe_generation", True)

    train_envs = build_env(config, config.num_envs)
    train_envs = VecNormalize(train_envs, norm_obs=False)

    eval_env = build_env(config, config.n_eval_envs)
    eval_env = VecNormalize(eval_env, training=False, norm_obs=False)
    eval_env.seed(config.validation_seed)

    best_model_save_path = str(run_model_path)
    callbacks = []
    if getattr(config, 'curriculum_learning', False):
        timesteps = getattr(config, 'curriculum_timesteps', [1e4, 3e4, 8e4, 2e5, 3.5e5, 5e5])
        gen_numbers = getattr(config, 'curriculum_gen_numbers', [4, 5, 6, 7, 9, 10])
        callbacks.append(AdaptGenNumber(timesteps, gen_numbers))
    if not config.disable_wandb:
        callbacks.append(WandbCallback())
    sync_stats_callback = SaveBestVecNormalizeCallback(
        save_dir=best_model_save_path
    )
    callbacks_after_eval = CallbackList(callbacks)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_save_path,
        callback_on_new_best=sync_stats_callback,
        log_path=str(run_log_path),
        eval_freq=int(config.eval_freq // config.num_envs),
        n_eval_episodes=int(config.n_eval_episodes),
        deterministic=True,
        render=False,
        callback_after_eval=callbacks_after_eval,
    )
    if config.observation == "het_interaction":
        from src.networks.het_interaction_selection_model import (
            HetCNNFeaturesExtractor,
            HetInteractionSelectionAC,
        )
        policy_kwargs = dict(
            features_extractor_class=HetCNNFeaturesExtractor,
            features_extractor_kwargs=dict(
                conv1_kwargs=conv1_kwargs,
                conv2_kwargs=conv2_kwargs,
                features_dim=64,
            ),
            policy_hiddens=int(config.policy_hiddens),
            value_hiddens=int(config.value_hiddens),
            gen_features_dim=int(config.gen_features_dim),
            het_features_dim=int(config.het_features_dim), #new
        )
    else:
        policy_kwargs = dict(
            features_extractor_class=CNNFeaturesExtractor,
            features_extractor_kwargs=dict(
                conv1_kwargs=conv1_kwargs,
                conv2_kwargs=conv2_kwargs,
                features_dim=64,
                use_gen_obs=use_gen_obs,
            ),
            policy_hiddens=int(config.policy_hiddens),
            value_hiddens=int(config.value_hiddens),
            gen_features_dim=int(config.gen_features_dim),
            use_gen_obs=use_gen_obs,
        )

    model = PPO(
        SelectionAC if config.observation != "het_interaction" else HetInteractionSelectionAC,
        train_envs,
        seed=int(config.seed),
        verbose=1,
        tensorboard_log=str(run_log_path),
        policy_kwargs=policy_kwargs,
        n_steps=int(config.n_steps),
        batch_size=int(config.batch_size),
        gamma=float(config.gamma),
        learning_rate=float(config.learning_rate),
        gae_lambda=float(config.gae_lambda),
        n_epochs=int(config.n_epochs),
        ent_coef=float(config.ent_coeff),
    )

    model.learn(
        total_timesteps=int(config.total_timesteps),
        callback=eval_callback,
    )

    model.save(run_model_path / f"{config.unique_name}_final")
    # train_envs.save(str(run_model_path / "vec_normalize_final.pkl"))
    # torch.save(model.policy.features_extractor.shared_network.state_dict(), str(run_model_path / "shared_weights.pth"),)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Config:        {config.config_path}")
    print(f"Training seed: {config.seed}")
    print(f"Run directory: {run_model_path}")
    print("=" * 80 + "\n")

    if not config.disable_wandb:
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    if args.seed is None:
        raise ValueError("--seed is required for training")
    config = load_config(args.config)

    # CLI overrides
    config.seed = int(args.seed)
    config.validation_seed = int(args.validation_seed)
    config.config_path = args.config

    from pathlib import Path
    config.log_path = Path(config.log_path)
    config.model_path = Path(config.model_path)

    main(config)