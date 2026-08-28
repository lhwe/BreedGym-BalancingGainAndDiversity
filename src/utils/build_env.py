
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from breedgym.utils.paths import DATA_PATH, MODEL_PATH, LOG_PATH
from breedgym.vector import VecBreedGym, SelectionScores
from src.wrappers.vector.adapt_vec_env import AdaptVecEnv
from src.wrappers.vector.gen_number_obs import ObserveGenNumber
from src.wrappers.vector.mask_snps import MaskSNPs
from src.wrappers.vector.max_gebv_reward import MaxGEBVRewardWrapper


def build_env(config, num_envs, method=None):
    """
    :param config: configuration yaml file for training or eval.
    :param num_envs: number of environments.
    :param method: dictionary of methods (used during eval; can be None during training)
    """
    env = VecBreedGym(
        num_envs=num_envs,
        initial_population=DATA_PATH.joinpath(config.genome),
        genetic_map=DATA_PATH.joinpath(config.genetic_map),
        individual_per_gen=config.individual_per_gen,
        reward_shaping=getattr(config, "reward_shaping", False),
        num_generations=int(config.num_generations),
        autoreset=False,
    )

    # prioritize using method dict, otherwise config
    def get_param(key, default=None):
        if method and key in method:
            return method[key]
        return getattr(config, key, default)

    reward_type = get_param("reward_type", "max_gebv")
    if reward_type == "full_sib":
        from src.wrappers.vector.pedigree_inbreeding_reward import FullSibPenaltyRewardWrapper
        env = FullSibPenaltyRewardWrapper(
            env,
            w_gain=get_param("w_gain", 1.0),
            w_sib_penalty=get_param("w_sib_penalty", 5.0))
    elif reward_type == "hybrid":
        from src.wrappers.vector.pedigree_inbreeding_reward import PedigreeInbreedingRewardWrapper
        env = PedigreeInbreedingRewardWrapper(
            env,
            w_gain=get_param("w_gain", 1.0),
            he_prop=get_param("he_prop", 0.3),
            w_sib_penalty=get_param("w_sib_penalty", 5.0),
            ema_alpha=get_param("ema_alpha", 0.1),
            min_he_weight=get_param("min_he_weight", 10.0),
            max_he_weight=get_param("max_he_weight", 500.0),
        )

    env = SelectionScores(env, k=config.k_best, n_crosses=config.n_crosses)

    if reward_type == "max_gebv":
        env = MaxGEBVRewardWrapper(
            env,
            w_gain=1.0,
            w_std=get_param("w_std", 0.0),
            w_inbreed=get_param("w_inbreed", 0.0))

    obs_type = get_param("observation", "standard")
    use_gen_obs = get_param("use_observe_generation", True)
    if obs_type == "standard":
        env = MaskSNPs(env)
        env = ObserveGenNumber(env, enabled=use_gen_obs)
    elif obs_type in ("het", "het_interaction"):
        from src.wrappers.vector.observe_heterozygosity import ObserveHeterozygosity
        from src.wrappers.vector.mask_snps_het import MaskSNPsHet
        from src.wrappers.vector.observe_gen_number_het import ObserveGenNumberHet
        if not config.use_heterozygosity_observation:
            raise ValueError(
                "Method requests heterozygosity observation, "
                "but config.use_heterozygosity_observation is False."
            )
        env = ObserveHeterozygosity(env)
        env = MaskSNPsHet(env)
        env = ObserveGenNumberHet(env)
    else:
        raise ValueError(f"Unknown observation type: {obs_type}")

    env = AdaptVecEnv(env)
    env = VecMonitor(env)
    return env

