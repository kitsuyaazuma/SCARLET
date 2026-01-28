# SCARLET
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fkitsuyaazuma%2FSCARLET%2Frefs%2Fheads%2Fmain%2Fpyproject.toml)
![GitHub License](https://img.shields.io/github/license/kitsuyaazuma/SCARLET)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/kitsuyaazuma/SCARLET/ci.yaml?label=CI)
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FTMC.2026.3652819-blue)](https://doi.org/10.1109/TMC.2026.3652819)
[![arXiv](https://img.shields.io/badge/arXiv-2504.19602-b31b1b.svg)](https://arxiv.org/abs/2504.19602)


Official implementation of SCARLET: "[Soft-Label Caching and Sharpening for Communication-Efficient Federated Distillation](https://ieeexplore.ieee.org/document/11344746)" (Accepted by IEEE TMC).

<p align="center">
  <img src="assets/overview.png" width="100%" alt="SCARLET Overview">
</p>

> [!IMPORTANT]
> The `main` branch contains a simplified implementation for better understanding of SCARLET’s core algorithms.
> For the exact experiment code and hyperparameter settings used in our paper, switch to the [`reproducibility`](https://github.com/kitsuyaazuma/SCARLET/tree/reproducibility) branch.

# Getting Started

## uv

```bash
git clone https://github.com/kitsuyaazuma/SCARLET.git
cd SCARLET
uv sync

uv run python -m scarlet.main scarlet
```

## Docker

```bash
docker run -it --rm --gpus=all --name scarlet ghcr.io/kitsuyaazuma/scarlet:main scarlet

# or

git clone https://github.com/kitsuyaazuma/SCARLET.git
cd SCARLET
docker build -t scarlet .
docker run -it --rm --gpus=all --name scarlet scarlet:latest scarlet
```

# Configuration

All hyperparameters are managed with [Tyro](https://github.com/brentyi/tyro). You can see all available options by running:

```bash
uv run python -m scarlet.main --help
```

# Citation

If you use this code in your research, please cite our preprint:

```bibtex
@ARTICLE{11344746,
  author={Azuma, Kitsuya and Nishio, Takayuki and Kitagawa, Yuichi and Nakano, Wakako and Tanimura, Takahito},
  journal={IEEE Transactions on Mobile Computing}, 
  title={Soft-Label Caching and Sharpening for Communication-Efficient Federated Distillation}, 
  year={2026},
  volume={},
  number={},
  pages={1-18},
  keywords={Servers;Computational modeling;Data models;Mobile computing;Entropy;Data privacy;Accuracy;Training;Quantization (signal);Federated learning;Federated learning;knowledge distillation;non-IID data;communication efficiency},
  doi={10.1109/TMC.2026.3652819}}
```
