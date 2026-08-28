import matplotlib.pyplot as plt
from utils.evaluation import *
from src.utils.config import load_config
from src.utils.args import parse_args

STD_COLORS = {
    0.0: "#d62728",   # red
    0.5: "#377eb8",   # blue
    1.0: "#2ca02c",   # green
}

HE_MARKERS = {
    0.0: "o",
    500.0: "s",
    1000.0: "^",
}

PEDIGREE_STYLES = {
    "Full-sib Penalty": {"color": "#9467bd", "marker": "D"},
    "Hybrid": {"color": "#ff7f0e", "marker": "D"},
    "Standard GS": {"color": "#222222", "marker": "o"},
    "Standard PPO": {"color": "#999999", "marker": "o"},
}


def _method_style(method_name, method_lookup, fallback_idx=0):
    """
    Determine the plotting style for a given method.

    Returns a dict with:
        - color: based on w_std value (or fallback color)
        - marker: based on w_He value (or fallback marker)
        - w_std: the w_std value if available
        - w_he: the w_he value if available
    """
    # Check if this is a pedigree/reference method with fixed style
    if method_name in PEDIGREE_STYLES:
        style = PEDIGREE_STYLES[method_name].copy()
        style['w_std'] = None
        style['w_he'] = None
        return style

    # Get method config
    method = method_lookup.get(method_name, {})

    # Determine w_std and w_He
    w_std = _get_w_std(method)
    w_he = _get_w_he(method)

    # Get color from STD_COLORS based on w_std
    if w_std is not None:
        nearest_key = _nearest_key(w_std, STD_COLORS)
        if nearest_key is not None:
            color = STD_COLORS[nearest_key]
        else:
            # Use a fallback color from the color cycle
            from matplotlib import rcParams
            color_cycle = rcParams["axes.prop_cycle"].by_key()["color"]
            color = color_cycle[fallback_idx % len(color_cycle)]
    else:
        # No w_std defined, use fallback
        from matplotlib import rcParams
        color_cycle = rcParams["axes.prop_cycle"].by_key()["color"]
        color = color_cycle[fallback_idx % len(color_cycle)]

    # Get marker from HE_MARKERS based on w_He
    if w_he is not None:
        nearest_key = _nearest_key(w_he, HE_MARKERS)
        if nearest_key is not None:
            marker = HE_MARKERS[nearest_key]
        else:
            # Use a fallback marker
            fallback_markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]
            marker = fallback_markers[fallback_idx % len(fallback_markers)]
    else:
        # No w_He defined, use default circle
        marker = "o"

    return {
        "color": color,
        "marker": marker,
        "w_std": w_std,
        "w_he": w_he,
    }


def _as_float(value):
    """Convert a numeric config value to float; return None if unavailable."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _get_w_std(method):
    """Read w_std from a method config, supporting int/float YAML values."""
    return _as_float(method.get("w_std"))


def _get_w_he(method):
    """Read w_He / w_he from a method config."""
    for key in ("w_He", "w_he", "w_heterozygosity", "w_inbreed", "inbreed_weight"):
        if key in method:
            return _as_float(method.get(key))
    return None


def _nearest_key(value, mapping, tol=1e-8):
    """Find an almost-equal numeric key in a small style mapping."""
    if value is None:
        return None
    for key in mapping:
        if abs(value - float(key)) <= tol:
            return key
    return None


def _get_terminal_seed_points(data):
    """
    Return one terminal observation per training seed/policy.

    Test trials and eval environments have already been averaged
    inside evaluate_rl_method().
    """
    x = np.asarray(data["gebv"][:, -1], dtype=float)
    y = np.asarray(data["diversity"][:, -1], dtype=float)
    return x, y

def _ellipse_from_points(
    ax,
    x,
    y,
    color,
    n_std=1.5,
    alpha=0.14,
    linewidth=1.6,
):
    """Draw a policy-specific covariance ellipse from its training seeds."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) < 2:
        return

    points = np.column_stack([x, y])
    center = np.median(points, axis=0)

    cov = np.cov(points, rowvar=False)
    if cov.shape != (2, 2):
        return

    cov = cov + np.eye(2) * 1e-12

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 1e-12)
    eigvecs = eigvecs[:, order]

    width = 2.0 * n_std * np.sqrt(eigvals[0])
    height = 2.0 * n_std * np.sqrt(eigvals[1])
    angle = np.degrees(
        np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
    )

    from matplotlib.patches import Ellipse

    ellipse = Ellipse(
        xy=center,
        width=width,
        height=height,
        angle=angle,
        facecolor=color,
        edgecolor="none",
        alpha=alpha,
        linewidth=linewidth,
        zorder=1,
    )
    ax.add_patch(ellipse)


def _draw_seed_halo(ax, x, y, style, seed, annotate_seeds=False):
    ax.scatter(
        x,
        y,
        s=150,
        marker="o",
        facecolor="none",
        edgecolor="#8c8c8c",
        linewidth=1.15,
        alpha=0.48,
        zorder=2,
    )

    # Parameter encoding: color=w_std, marker=w_He
    ax.scatter(
        x,
        y,
        s=52,
        marker=style["marker"],
        color=style["color"],
        edgecolor="none",
        linewidth=0,
        alpha=0.96,
        zorder=4,
    )


def _legend_color_handles():
    from matplotlib.lines import Line2D

    return [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor=color,
            markeredgecolor="none",
            markersize=7.5,
            label=f"w_std={value:g}",
        )
        for value, color in STD_COLORS.items()
    ]


def _legend_marker_handles():
    from matplotlib.lines import Line2D

    return [
        Line2D(
            [0], [0],
            marker=marker,
            linestyle="None",
            markerfacecolor="#777777",
            markeredgecolor="none",
            markersize=7.5,
            label=f"w_He={value:g}",
        )
        for value, marker in HE_MARKERS.items()
    ]


def _add_parameter_legends(ax, show_seed_halo=False):
    """
    Two compact legends matching the requested visual grammar:
        Color -> w_std
        Marker -> w_He
    """
    from matplotlib.lines import Line2D

    # Color legend (w_std)
    color_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            color=color,
            markeredgecolor="black",
            markeredgewidth=0.5,
            markersize=8,
            label=f"w_std={value:g}",
        )
        for value, color in STD_COLORS.items()
    ]

    color_legend = ax.legend(
        handles=color_handles,
        title="Color: w_std",
        loc="upper left",
        fontsize=8,
        title_fontsize=8,
        frameon=True,
        framealpha=0.92,
        borderpad=0.6,
        handletextpad=0.5,
        labelspacing=0.35,
    )
    ax.add_artist(color_legend)

    # Marker legend (w_He)
    marker_handles = [
        Line2D(
            [0], [0],
            marker=marker,
            linestyle="",
            color="white",
            markeredgecolor="black",
            markeredgewidth=0.5,
            markersize=8,
            label=f"w_He={value:g}",
        )
        for value, marker in HE_MARKERS.items()
    ]

    if show_seed_halo:
        marker_handles.append(
            Line2D(
                [0], [0],
                marker="o",
                linestyle="",
                markerfacecolor="none",
                markeredgecolor="#8c8c8c",
                markersize=10,
                label="Seed halo",
            )
        )

    ax.legend(
        handles=marker_handles,
        title="Marker: w_He",
        loc="lower right",
        fontsize=8,
        title_fontsize=8,
        frameon=True,
        framealpha=0.92,
        borderpad=0.6,
        handletextpad=0.5,
        labelspacing=0.35,
    )


def _setup_panel(ax, title):
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Final max GEBV", fontsize=11)
    ax.set_ylabel("Final expected heterozygosity", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)


def _annotate_pedigree_method(ax, reward_name, x, y):
    """Use the compact labels shown in the reference figure."""
    labels = {
        "Full-sib Penalty": "Full-sib",
        "Hybrid": "Hybrid",
        "Standard GS": "Max GEBV",
    }
    label = labels.get(reward_name, reward_name)

    dx = 8
    dy = 7
    if reward_name == "Standard GS":
        dx, dy = 8, 8

    ax.annotate(
        label,
        (x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=9.5,
        color="#333333",
    )


def _draw_seed_level_panel(
        ax,
        panel_methods,
        results,
        method_lookup,
        title,
        annotate_seeds=False,
):
    """
    Version 1: every training seed is shown.

    Encoding:
        color = w_std
        marker = w_He
        neutral outer ring = seed
    """
    has_scalarized_params = False

    for reward_name in panel_methods:
        data = results[reward_name]
        x, y = _get_terminal_seed_points(data)
        style = _method_style(
            reward_name,
            method_lookup,
            fallback_idx=panel_methods.index(reward_name),
        )

        # Check if this method has scalarized parameters
        if style.get('w_std') is not None or style.get('w_he') is not None:
            has_scalarized_params = True

        if reward_name == "Standard GS":
            ax.scatter(
                x[0],
                y[0],
                s=95,
                marker=style["marker"],
                color="#222222",
                edgecolor="none",
                linewidth=1.0,
                zorder=5,
            )
            _annotate_pedigree_method(ax, reward_name, x[0], y[0])
            continue

        seeds = np.asarray(data["seeds"], dtype=int)
        for seed, x_value, y_value in zip(seeds, x, y):
            _draw_seed_halo(
                ax,
                x_value,
                y_value,
                style,
                seed,
                annotate_seeds=annotate_seeds,
            )

    _setup_panel(ax, title)

    # Add parameter legends only if there are scalarized methods in this panel
    if has_scalarized_params:
        _add_parameter_legends(ax, show_seed_halo=True)


def _draw_median_region_panel(
    ax,
    panel_methods,
    results,
    method_lookup,
    title,
    draw_ellipse=True,
):
    """
    Version 2/3: one median point per policy, Version 2 adds seed ellipse.
    """
    for reward_name in panel_methods:
        data = results[reward_name]
        x, y = _get_terminal_seed_points(data)

        style = _method_style(
            reward_name,
            method_lookup,
            fallback_idx=panel_methods.index(reward_name),
        )
        color = "#222222" if reward_name == "Standard GS" else style["color"]

        median_x = float(x[0]) if reward_name == "Standard GS" else float(np.median(x))
        median_y = float(y[0]) if reward_name == "Standard GS" else float(np.median(y))

        if draw_ellipse and reward_name != "Standard GS":
            _ellipse_from_points(
                ax,
                x,
                y,
                color=color,
                n_std=1.5,
                alpha=0.14,
                linewidth=1.6,
            )

        ax.scatter(
            median_x,
            median_y,
            s=125,
            marker=style["marker"],
            color=color,
            edgecolor="none",
            linewidth=1.25,
            zorder=5,
        )

        ax.scatter(
            median_x,
            median_y,
            s=125,
            marker=style["marker"],
            facecolor="none",
            edgecolor="#333333",
            linewidth=0.55,
            zorder=6,
        )

        if reward_name in {"Full-sib Penalty", "Hybrid", "Standard GS"}:
            _annotate_pedigree_method(ax, reward_name, median_x, median_y)

    _setup_panel(ax, title)

    if any(
        _method_style(name, method_lookup).get("w_std") is not None
        for name in panel_methods
    ):
        _add_parameter_legends(ax, show_seed_halo=False)


def _split_methods_for_reward_panels(results, method_lookup):
    """
    Split methods into:
        (a) scalarized reward tuning
        (b) pedigree-aware reward

    Standard GS is included in both panels as the common baseline.
    """
    scalarized = []
    pedigree = []

    for reward_name in results.keys():
        if reward_name == "Standard GS":
            continue

        method = method_lookup.get(reward_name, {})
        reward_type = method.get("reward_type", "")

        if reward_type == "max_gebv":
            scalarized.append(reward_name)
        elif reward_type in {"full_sib", "hybrid"}:
            pedigree.append(reward_name)
        else:
            name_lower = reward_name.lower()
            if any(
                token in name_lower
                for token in ("pedigree", "full-sib", "full sib", "hybrid", "sib")
            ):
                pedigree.append(reward_name)
            else:
                scalarized.append(reward_name)

    if "Standard GS" in results:
        scalarized.append("Standard GS")
        pedigree.append("Standard GS")

    return scalarized, pedigree


def plot_reward_figure(
    results,
    config,
    save_path,
    version=1,
):
    """
    Create one two-panel reward figure.

    Version 1: seed-level points + neutral seed halo.
    Version 2: policy median + covariance ellipse
    Version 3: median points only, no ellipse.
    """
    if not config.monitor_heterozygosity:
        raise ValueError(
            "Expected heterozygosity is required for the requested figure."
        )

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    method_lookup = {
        method["name"]: method
        for method in config.methods
    }

    scalarized_methods, pedigree_methods = _split_methods_for_reward_panels(
        results,
        method_lookup,
    )

    color_cycle = plt.rcParams[
        "axes.prop_cycle"
    ].by_key()["color"]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.2, 5.8),
    )

    ax_a, ax_b = axes

    if version == 1:
        _draw_seed_level_panel(
            ax_a,
            scalarized_methods,
            results,
            method_lookup,
            "(A) Scalarized reward tuning",
            # color_cycle,
            annotate_seeds=getattr(config, "annotate_seeds", False),
        )
        _draw_seed_level_panel(
            ax_b,
            pedigree_methods,
            results,
            method_lookup,
            "(B) Pedigree-aware reward",
            # color_cycle,
            annotate_seeds=getattr(config, "annotate_seeds", False),
        )

    elif version == 2:
        _draw_median_region_panel(
            ax_a,
            scalarized_methods,
            results,
            method_lookup,
            "(A) Scalarized reward tuning",
            # color_cycle,
            draw_ellipse=True,
        )
        _draw_median_region_panel(
            ax_b,
            pedigree_methods,
            results,
            method_lookup,
            "(B) Pedigree-aware reward",
            # color_cycle,
            draw_ellipse=True,
        )

    elif version == 3:
        _draw_median_region_panel(
            ax_a,
            scalarized_methods,
            results,
            method_lookup,
            "(A) Scalarized reward tuning",
            # color_cycle,
            draw_ellipse=False,
        )
        _draw_median_region_panel(
            ax_b,
            pedigree_methods,
            results,
            method_lookup,
            "(B) Pedigree-aware reward",
            # color_cycle,
            draw_ellipse=False,
        )

    else:
        raise ValueError(f"Unknown plotting version: {version}")

    fig.tight_layout(pad=1.2)

    fig.savefig(
        save_path,
        dpi=350,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    print(f">>> Saved reward figure Version {version}: {save_path}")

def main(config):
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.cache_dir, exist_ok=True)

    results = {}
    experiment_name = getattr(config, 'experiment_name', 'reward_comp')

    # Methods
    for method in config.methods:
        method_name = method["name"]
        if method.get("baseline", False):
            result = evaluate_standard_gs(config)
        else:
            result = evaluate_rl_method(config, method)
        results[method_name] = result

    # Report
    print_final_report(
        results,
        config.num_generations,
        monitor_heterozygosity=config.monitor_heterozygosity,
        interval=config.interval,
    )

    # Save cache
    np.savez(
        os.path.join(config.cache_dir, f"{experiment_name}_results.npz"),
        **{f"{method}_gebv": data["gebv"] for method, data in results.items()},
        **{
            f"{method}_he": data["diversity"]
            for method, data in results.items()
            if "diversity" in data
        },
    )

    # Plot: generate all three requested visual versions.
    seed_halo_path = os.path.join(
        config.save_dir,
        f"{experiment_name}_gen_{config.num_generations}"
        f"_seed_halo.png",
    )

    median_halo_path = os.path.join(
        config.save_dir,
        f"{experiment_name}_gen_{config.num_generations}"
        f"_median_halo.png",
    )

    no_halo_path = os.path.join(
        config.save_dir,
        f"{experiment_name}_gen_{config.num_generations}"
        f"_no_halo.png",
    )

    # Version 1: seed-level points + neutral seed halo.
    plot_reward_figure(
        results,
        config,
        save_path=seed_halo_path,
        version=1,
    )

    # Version 2: policy median + covariance ellipse
    plot_reward_figure(
        results,
        config,
        save_path=median_halo_path,
        version=2,
    )

    # Version 3: policy median only
    plot_reward_figure(
        results,
        config,
        save_path=no_halo_path,
        version=3,
    )

    # CSV
    save_terminal_csv(
        results,
        os.path.join(config.save_dir, f"{experiment_name}_terminal_results.csv"),
    )

    print("\n>>> Evaluation complete.")


def main_with_npz(config, npz_path):
    results = load_results_from_npz(npz_path, config.seeds)

    if results:
        print_final_report(
            results,
            config.num_generations,
            monitor_heterozygosity=config.monitor_heterozygosity,
            interval=config.interval,
        )

    os.makedirs(config.save_dir, exist_ok=True)
    experiment_name = getattr(config, 'experiment_name', 'reward_figure')

    seed_level_path = os.path.join(
        config.save_dir,
        f"{experiment_name}_gen_{config.num_generations}_seed_level.png"
    )

    median_region_path = os.path.join(
        config.save_dir,
        f"{experiment_name}_gen_{config.num_generations}_median_region.png"
    )

    plot_reward_figure(
        results,
        config,
        save_path=seed_level_path,
        version=1,
    )

    plot_reward_figure(
        results,
        config,
        save_path=median_region_path,
        version=2,
    )

    print(f"\n>>> Evaluation complete. Figures saved to {config.save_dir}")

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

    # main_with_npz(config, "figures/caches/reward_figure_results.npz")
    main(config)