import argparse


def parse_seeds(value):
    """Parse comma-separated seed list."""
    seeds = tuple(
        int(seed.strip())
        for seed in value.split(",")
        if seed.strip()
    )
    if not seeds:
        raise argparse.ArgumentTypeError("at least one training seed is required")
    return seeds

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/eval/reward_adjustw.yaml",
        )

    parser.add_argument(
        "--num-generations",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--test-seed-start",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Single training seed (for training scripts)"
    )

    parser.add_argument(
        "--validation-seed",
        type=int,
        default=12345,
        help="Validation seed (for training scripts)"
    )

    parser.add_argument(
        "--seeds",
        type=parse_seeds,
        default=None,
        help="Comma-separated list of training seeds (for eval scripts)"
    )

    parser.add_argument(
        "--annotate-seeds",
        action="store_true",
    )

    return parser.parse_args()