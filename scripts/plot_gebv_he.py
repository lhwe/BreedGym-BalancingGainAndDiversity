import matplotlib.pyplot as plt
from utils.evaluation import *
from src.utils.config import load_config
from src.utils.args import parse_args


METHOD_STYLES = {
    "Generation-aware reference": {
        "color": "#e41a1c",
        "linestyle": "-",
        "marker": "s",
    },
    "Heterozygosity-augmented": {
        "color": "#377eb8",
        "linestyle": "-",
        "marker": "s",
    },
    "Standard GS": {
        "color": "#000000",
        "linestyle": "--",
        "marker": "o",
    },
    "RL: Vanilla": {
        "color": "#1f77b4",
        "linestyle": "-",
        "marker": "s",
    },
    "RL: Observe Generation": {
        "color": "#d62728",
        "linestyle": "-",
        "marker": "s",
    },
    "RL: Curriculum Learning": {
        "color": "#2ca02c",
        "linestyle": "-",
        "marker": "s",
    },
}

# Plot
def plot_main_figure(
    results,
    num_generations,
    interval=90,
    monitor_heterozygosity=True,
    save_path="figures/obs_compare.png",
):
    print(
        f">>> Generating main figure -> "
        f"{save_path}"
    )

    os.makedirs(
        os.path.dirname(save_path)
        or ".",
        exist_ok=True,
    )

    gens = np.arange(1, num_generations + 1,)

    if monitor_heterozygosity:
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5),)
        ax_gebv, ax_he = axes
    else:
        fig, ax_gebv = plt.subplots(figsize=(6.0, 4.5))
        ax_he = None

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_idx = 0

    for reward_name, data in results.items():
        style = METHOD_STYLES.get(
            reward_name,
            {
                "color": color_cycle[color_idx % len(color_cycle)],
                "linestyle": "-",
                "marker": "o",
            },
        )
        color_idx += 2

        gebv_med, gebv_low, gebv_high = (
            interval_bounds(
                data["gebv"][:, 1:],
                interval=interval,
                axis=0,
            )
        )

        ax_gebv.fill_between(
            gens,
            gebv_low,
            gebv_high,
            color=style["color"],
            alpha=0.18,
            linewidth=0,
        )

        ax_gebv.plot(
            gens,
            gebv_med,
            label=reward_name,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=2.0,
            markersize=4,
        )

        if monitor_heterozygosity:
            he_med, he_low, he_high = (
                interval_bounds(
                    data["diversity"][:, 1:],
                    interval=interval,
                    axis=0,
                )
            )

            ax_he.fill_between(
                gens,
                he_low,
                he_high,
                color=style["color"],
                alpha=0.18,
                linewidth=0,
            )

            ax_he.plot(
                gens,
                he_med,
                label=reward_name,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=2.0,
                markersize=4,
            )

    if num_generations <= 10:
        x_ticks = np.arange(1, num_generations + 1)

    else:
        x_ticks = np.arange(5, num_generations + 1, 5)

        # Always show the final generation, even if it is not
        # a multiple of 5.
        if num_generations not in x_ticks:
            x_ticks = np.append(x_ticks, num_generations)

        x_ticks = np.unique(x_ticks)

        # Vertical dashed line at generation 10
        axes_to_use = (
            [ax_gebv, ax_he]
            if monitor_heterozygosity
            else [ax_gebv]
        )

        for axis in axes_to_use:
            if axis is not None:
                axis.axvline(
                    10,
                    color="#666666",
                    linestyle="--",
                    linewidth=1.2,
                    zorder=0,
                )

    ax_gebv.set_title(
        "(a) Maximum GEBV",
        fontsize=12,
        fontweight="bold",
    )

    ax_gebv.set_xlabel(
        "Generation",
        fontsize=11,
    )

    ax_gebv.set_ylabel(
        "Maximum GEBV",
        fontsize=11,
    )

    ax_gebv.set_xticks(x_ticks)
    ax_gebv.grid(
        True,
        linestyle="--",
        alpha=0.5,
    )
    ax_gebv.legend(
        loc="lower right",
        fontsize=9,
    )

    if monitor_heterozygosity:
        ax_he.set_title(
            "(b) Expected heterozygosity",
            fontsize=12,
            fontweight="bold",
        )

        ax_he.set_xlabel(
            "Generation",
            fontsize=11,
        )

        ax_he.set_ylabel(
            "Expected heterozygosity",
            fontsize=11,
        )

        ax_he.set_xticks(x_ticks)

        ax_he.grid(
            True,
            linestyle="--",
            alpha=0.5,
        )

        ax_he.legend(
            loc="upper right",
            fontsize=9,
        )

    fig.tight_layout()

    fig.savefig(
        save_path,
        dpi=300,
        # bbox_inches="tight",
    )

    plt.close(fig)

    print(">>> Main figure complete.")

# Terminal scatter
def plot_terminal_seed_scatter(
    results,
    save_path,
    annotate_seeds=False,
):
    print(
        f">>> Generating terminal scatter -> "
        f"{save_path}"
    )

    os.makedirs(
        os.path.dirname(save_path)
        or ".",
        exist_ok=True,
    )

    fig, ax = plt.subplots(
        figsize=(6.2, 5.2)
    )

    for reward_name, data in results.items():
        style = METHOD_STYLES.get(
            reward_name,
            {
                "color": None,
            },
        )

        terminal_he = data[
            "diversity"
        ][:, -1]

        terminal_gebv = data[
            "gebv"
        ][:, -1]

        ax.scatter(
            terminal_he,
            terminal_gebv,
            s=55,
            color=style["color"],
            edgecolor="white",
            linewidth=0.7,
            alpha=0.9,
            label=reward_name,
        )

        ax.scatter(
            np.median(terminal_he),
            np.median(terminal_gebv),
            s=120,
            marker="X",
            color=style["color"],
            edgecolor="black",
            linewidth=0.8,
            zorder=4,
        )

        if annotate_seeds:
            for seed, x_value, y_value in zip(
                data["seeds"],
                terminal_he,
                terminal_gebv,
            ):

                ax.annotate(
                    str(seed),
                    (x_value, y_value),
                    xytext=(4, 3),
                    textcoords="offset points",
                    fontsize=7,
                    color=style["color"],
                )

    ax.set_xlabel(
        "Terminal expected heterozygosity",
        fontsize=11,
    )

    ax.set_ylabel(
        "Terminal maximum GEBV",
        fontsize=11,
    )

    ax.set_title(
        "Terminal outcomes across training seeds",
        fontsize=12,
    )

    ax.grid(
        True,
        linestyle="--",
        alpha=0.5,
    )

    ax.legend(
        fontsize=9
    )

    fig.tight_layout()

    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        ">>> Terminal scatter complete."
    )

def main(config):
    os.makedirs(config.save_dir, exist_ok=True,)
    os.makedirs(config.cache_dir, exist_ok=True,)

    results = {}
    experiment_name = getattr(config, 'experiment_name', 'unknown_compare')

    # Methods
    for method in config.methods:
        method_name = method["name"]
        if method.get("baseline",False,):
            result = evaluate_standard_gs(config)
        else:
            result = evaluate_rl_method(config,method,)

        results[method_name] = result

    # Report
    print_final_report(
        results,
        config.num_generations,
        monitor_heterozygosity=(
            config.monitor_heterozygosity
        ),
        interval=config.interval,
    )

    # Save cache
    np.savez(
        os.path.join(
            config.cache_dir,
            f"{experiment_name}_results.npz",
        ),
        **{
            f"{method}_gebv": data["gebv"]
            for method, data in results.items()
        },
        **{
            f"{method}_he": data["diversity"]
            for method, data in results.items()
            if "diversity" in data
        },
    )

    # Figures
    main_figure = os.path.join(
        config.save_dir,
        f"{experiment_name}_gen_{config.num_generations}.png",
    )

    plot_main_figure(
        results,
        config.num_generations,
        interval=config.interval,
        monitor_heterozygosity=(
            config.monitor_heterozygosity
        ),
        save_path=main_figure,
    )

    if config.monitor_heterozygosity:

        scatter_path = os.path.join(
            config.save_dir,
            f"{experiment_name}_terminal_seed_scatter_gen_{config.num_generations}.png",
        )

        plot_terminal_seed_scatter(
            results,
            scatter_path,
            annotate_seeds=(
                config.annotate_seeds
            ),
        )

    # CSV
    save_terminal_csv(
        results,
        os.path.join(
            config.save_dir,
            f"{experiment_name}_terminal_results.csv",
        ),
    )

    print(
        "\n>>> Evaluation complete."
    )


if __name__ == "__main__":

    args = parse_args()
    if args.seeds is None:
        raise ValueError("--seeds is required for eval")
    config = load_config(args.config)

    # CLI overrides
    if args.num_generations is not None:
        config.num_generations = args.num_generations
    if args.trials is not None:
        config.trials = args.trials
    if args.interval is not None:
        config.interval = args.interval
    if args.test_seed_start is not None:
        config.test_seed_start = args.test_seed_start
    if args.seeds is not None:
        config.seeds = args.seeds
    if args.annotate_seeds:
        config.annotate_seeds = True

    main(config)