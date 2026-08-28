import numpy as np


def individual_heterozygosity(obs):
    """
    Per-individual observed heterozygosity: fraction of loci where the two
    alleles differ, for each individual.

    Args:
        obs: raw genotype array, shape (num_envs, individuals, markers, 2)
    Returns:
        het: shape (num_envs, individuals), values in [0, 1]
    """
    het_sites = (obs[..., 0] != obs[..., 1])
    return np.mean(het_sites, axis=-1).astype(np.float32)