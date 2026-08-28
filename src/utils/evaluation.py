import csv
import os
from pathlib import Path
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecNormalize
from src.utils.build_env import build_env


def find_base_env(envs):
    current_layer = envs
    while current_layer is not None:
        if type(current_layer).__name__ == "VecBreedGym":
            return current_layer
        if hasattr(current_layer, "venv"):
            current_layer = current_layer.venv
        elif hasattr(current_layer, "gym_vec_env"):
            current_layer = current_layer.gym_vec_env
        elif hasattr(current_layer, "env"):
            current_layer = current_layer.env
        else:
            current_layer = None
    raise RuntimeError("Core VecBreedGym layer could not be found.")

def expected_heterozygosity(population):
    population = np.asarray(population)
    if population.ndim != 3 or population.shape[-1] != 2:
        raise ValueError(f"expected population shape (individuals, markers, 2), found {population.shape}")
    p = population.mean(axis=(0, 2))
    return float((2.0 * p * (1.0 - p)).mean())

def population_max_gebv(base_env):
    gebv = np.asarray(base_env.simulator.GEBV_model(base_env.populations))
    return gebv.max(axis=tuple(range(1, gebv.ndim)))

def evaluate_rl_method(config, method):
    """
    Evaluate one RL method across:
        - multiple independently trained policies
        - multiple matched test trials

    Output shape:
        gebv: (n_seeds, n_generations + 1)
        diversity: (n_seeds, n_generations + 1)
    """

    method_name = method["name"]
    seeds = tuple(config.seeds)

    seed_gebv = []
    seed_he = []

    print("\n" + "=" * 80)
    print(f"Evaluating: {method_name}")
    print("=" * 80)

    for training_seed in seeds:
        run_dir = Path(method["run_dir_template"].format(seed=training_seed))
        model_path = (run_dir / "best_model.zip")
        pkl_path = (run_dir / "vec_normalize.pkl")

        missing = [path for path in (model_path, pkl_path) if not path.is_file()]
        if missing:
            missing_text = "\n".join(f"  - {path}" for path in missing)
            raise FileNotFoundError(f"Missing files for {method_name}, training seed {training_seed}:\n{missing_text}")
        print(f"\n    training seed = {training_seed}")
        print(f"    run directory = {run_dir}")

        envs = build_env(config, config.n_eval_envs, method)
        try:
            envs = VecNormalize.load(str(pkl_path), envs)
            envs.training = False
            envs.norm_reward = False

            model = PPO.load(str(model_path))
            base_env = find_base_env(envs)

            buffer_gebv = np.zeros((config.trials, config.n_eval_envs, config.num_generations + 1))
            buffer_he = np.zeros_like(buffer_gebv)

            # Test trials
            for trial_idx in range(config.trials):
                test_seed = int(config.test_seed_start) + trial_idx

                set_random_seed(test_seed)
                envs.seed(test_seed)
                obs = envs.reset()

                # Initial GEBV
                buffer_gebv[trial_idx, :, 0] = population_max_gebv(base_env)

                # Initial HE
                if config.monitor_heterozygosity:
                    for env_idx in range(config.n_eval_envs):
                        buffer_he[trial_idx, env_idx, 0] = expected_heterozygosity(base_env.populations[env_idx])

                # Generations
                for gen in range(config.num_generations):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, _, _ = envs.step(action)
                    buffer_gebv[trial_idx, :, gen + 1] = population_max_gebv(base_env)
                    if config.monitor_heterozygosity:
                        for env_idx in range(config.n_eval_envs):
                            buffer_he[trial_idx, env_idx, gen + 1] = expected_heterozygosity(base_env.populations[env_idx])

        finally:
            envs.close()

        # Average eval trials first, then treat each training seed as one observation
        seed_gebv.append(buffer_gebv.mean(axis=(0, 1)))
        if config.monitor_heterozygosity:
            seed_he.append(buffer_he.mean(axis=(0, 1)))

        print(f"    finished training seed {training_seed}")

    result = {
        "seeds": np.asarray(seeds, dtype=int),
        "gebv": np.stack(seed_gebv),
    }

    if config.monitor_heterozygosity:
        result["diversity"] = np.stack(seed_he)

    return result

def evaluate_standard_gs(config):
    print("\n" + "=" * 80)
    print("Evaluating: Standard GS")
    print("=" * 80)

    method = {
        "name": "Standard GS",
        "baseline": True,
        "reward_type": "max_gebv",
        "observation": "standard",
        "use_observe_generation": False,
    }

    envs = build_env(config, config.n_eval_envs, method)
    base_env = find_base_env(envs)

    buffer_gebv = np.zeros(
        (
            config.trials,
            config.n_eval_envs,
            config.num_generations + 1,
        )
    )

    buffer_he = np.zeros_like(buffer_gebv)

    try:
        for trial_idx in range(config.trials):
            test_seed = int(config.test_seed_start) + trial_idx
            set_random_seed(test_seed)
            envs.seed(test_seed)
            obs = envs.reset()

            # Initial GEBV
            buffer_gebv[trial_idx, :, 0] = population_max_gebv(base_env)

            # Initial HE
            if config.monitor_heterozygosity:
                for env_idx in range(config.n_eval_envs):
                    buffer_he[trial_idx, env_idx, 0] = expected_heterozygosity(
                        base_env.populations[env_idx]
                    )

            # Generations
            for gen in range(config.num_generations):
                current_geno = obs["obs"] if isinstance(obs, dict) else obs
                gebvs = current_geno.sum(axis=(2, 3))

                # select highest GEBV
                action = gebvs
                obs, _, _, _ = envs.step(action)
                buffer_gebv[trial_idx, :, gen + 1] = population_max_gebv(base_env)

                if config.monitor_heterozygosity:
                    for env_idx in range(config.n_eval_envs):
                        buffer_he[trial_idx, env_idx, gen + 1] = expected_heterozygosity(
                            base_env.populations[env_idx]
                        )

    finally:
        envs.close()

    result = {
        "seeds": np.asarray([0], dtype=int),
        "gebv": buffer_gebv.mean(axis=(0, 1))[None, :],
    }

    if config.monitor_heterozygosity:
        result["diversity"] = buffer_he.mean(axis=(0, 1))[None, :]

    return result

def interval_bounds(values, interval=90, axis=0):
    tail = (100.0 - interval) / 200.0
    median = np.median(values, axis=axis)
    lower = np.quantile(values, tail, axis=axis)
    upper = np.quantile(values, 1.0 - tail, axis=axis)
    return median, lower, upper

def print_final_report(results, num_generations, monitor_heterozygosity=True, interval=90):
    reward_names = list(results.keys())
    baseline_name = reward_names[0]
    baseline = results[baseline_name]

    base_gebv_final = np.median(baseline["gebv"][:, -1])

    if monitor_heterozygosity:
        base_he_final = np.median(baseline["diversity"][:, -1])

    print("\n" + "=" * 100)
    print(f"FINAL REPORT, baseline = {baseline_name}")
    print("=" * 100)

    for name, data in results.items():
        gebv_final, gebv_low, gebv_high = interval_bounds(
            data["gebv"][:, -1],
            interval=interval,
        )

        gebv_imp = 100.0 * (gebv_final - base_gebv_final) / base_gebv_final

        print(
            f"[{name}] "
            f"Gen {num_generations}: "
            f"Max GEBV={gebv_final:.4f} "
            f"[{gebv_low:.4f}, {gebv_high:.4f}] "
            f"({gebv_imp:+.2f}%)"
        )

        if monitor_heterozygosity:
            he_final, he_low, he_high = interval_bounds(
                data["diversity"][:, -1],
                interval=interval,
            )

            he_imp = 100.0 * (he_final - base_he_final) / base_he_final

            he_initial = np.median(data["diversity"][:, 0])
            he_retention = 100.0 * he_final / he_initial

            print(
                f"             HE={he_final:.6f} "
                f"[{he_low:.6f}, {he_high:.6f}] "
                f"({he_imp:+.2f}%)"
            )
            print(f"             HE retention={he_retention:.2f}%")

    print(f"\nBrackets show the central {interval}% interval across training seeds.")
    print("=" * 100 + "\n")

def save_terminal_csv(results, save_path):
    rows = []

    for method_name, data in results.items():
        for idx, seed in enumerate(data["seeds"]):
            row = {
                "Method": method_name,
                "Training Seed": int(seed),
                "Final Max GEBV": float(data["gebv"][idx, -1]),
            }

            if "diversity" in data:
                row["Initial HE"] = float(data["diversity"][idx, 0])
                row["Final HE"] = float(data["diversity"][idx, -1])
                row["HE Retention (%)"] = (
                    100.0
                    * data["diversity"][idx, -1]
                    / data["diversity"][idx, 0]
                )

            rows.append(row)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    fieldnames = list(rows[0].keys())

    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f">>> Saved terminal CSV: {save_path}")


def load_results_from_npz(npz_path, seeds):
    npz_data = np.load(npz_path, allow_pickle=True)
    results = {}
    original_seeds = np.asarray(seeds, dtype=int)

    gebv_keys = [
        key for key in npz_data.keys()
        if key.endswith("_gebv")
    ]

    for gebv_key in gebv_keys:
        method_name = gebv_key[:-5]  # remove "_gebv"
        gebv = np.asarray(npz_data[gebv_key])

        if method_name == "Standard GS":
            method_seeds = np.asarray([0], dtype=int)
        else:
            if len(original_seeds) != gebv.shape[0]:
                raise ValueError(
                    f"Training seed count mismatch for "
                    f"{method_name}: "
                    f"config has {len(original_seeds)} seeds, "
                    f"but GEBV has {gebv.shape[0]} rows."
                )

            method_seeds = original_seeds.copy()

        results[method_name] = {
            "seeds": method_seeds,
            "gebv": gebv,
        }

        he_key = f"{method_name}_he"

        if he_key in npz_data:
            results[method_name]["diversity"] = np.asarray(npz_data[he_key])

        print(
            f"Loaded: {method_name:45s} "
            f"GEBV={gebv.shape}"
        )

    return results
