# BreedGym-BalancingGainAndDiversity

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg?logo=python&logoColor=white)](#installation)
[![Package Manager: uv](https://img.shields.io/badge/package_manager-uv-DE5FE9.svg?logo=astral&logoColor=white)](#2-create-a-virtual-environment)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-orange.svg)](LICENSE)

## Overview

This project builds on BreedGym and ChromaX to support experiments on genomic selection with reinforcement learning. It includes:

- Simulations with [BreedGym](https://github.com/younik/breedgym.git)
- Training scripts for RL-based genomic selection policies using PPO (Stable-Baselines3)
- Support for generation-aware and heterozygosity-augmented observations
- Diversity-aware reward functions
- Evaluation scripts and extended-horizon rollouts

## Installation

This project should be installed with **Python 3.12**.

### 1. Clone this repository

```bash
git clone https://github.com/lhwe/BreedGym-BalancingGainAndDiversity.git
cd BreedGym-BalancingGainAndDiversity
```

### 2. Create a virtual environment
Using [uv](https://docs.astral.sh/uv/):
```bash
uv venv --python 3.12
source .venv/bin/activate
```

### 3. Install dependencies
From the root of this repository:
```bash
uv sync
```

### 4. Clone BreedGym

First, clone the [BreedGym repository](https://github.com/younik/breedgym.git):

```bash
git clone https://github.com/younik/breedgym.git
cd breedgym
uv pip install -e .
```

## Quick start

An example command to train a generation-aware policy with the default GEBV reward:

```bash
python -m scripts.train \
    --config configs/train/gen_aware.yaml \
    --seed 2026
```

To evaluate a trained policy and plot results:

```bash
python -u -m scripts.plot_gebv_he \
    --config configs/evaluation/obs.yaml \
    --num-generations 30 \
    --trials 100
```