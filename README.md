# SCARLET
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fkitsuyaazuma%2FSCARLET%2Frefs%2Fheads%2Fmain%2Fpyproject.toml)
![GitHub License](https://img.shields.io/github/license/kitsuyaazuma/SCARLET)


Official implementation of SCARLET (Semi-supervised federated distillation with global CAching and Reduced soft-Label EnTropy)

> [!IMPORTANT]
> The `main` branch contains a simplified implementation for better understanding of SCARLET’s core algorithms.
> For the exact experiment code and hyperparameter settings used in our paper, switch to the [`reproducibility`](https://github.com/kitsuyaazuma/SCARLET/tree/reproducibility) branch.

# Getting Started

## uv

```bash
git clone https://github.com/kitsuyaazuma/SCARLET.git
cd SCARLET
uv sync

uv run python main.py +algorithm=scarlet
```

## Docker

```bash
docker run -it --rm --gpus=all --name scarlet ghcr.io/kitsuyaazuma/scarlet:main +algorithm=scarlet

# or

git clone https://github.com/kitsuyaazuma/SCARLET.git
cd SCARLET
docker build -t scarlet .
docker run -it --rm --gpus=all --name scarlet scarlet:latest +algorithm=scarlet
```

# Configuration

All hyperparameters live under `config/` and are managed with [Hydra](https://github.com/facebookresearch/hydra). You can override any setting on the command line:

```bash
uv run python main.py \
    dir_alpha=0.05 \
    num_parallels=10 \
    +algorithm=scarlet \
    algorithm.enhanced_era_exponent=2.0 \
    algorithm.cache_duration=50
```

# Citation

If you use this code, please cite our manuscript:

```bibtex
@unpublished{azuma2025scarlet,
  author       = {Azuma, Kitsuya and Nishio, Takayuki and Kitagawa, Yuichi and Nakano, Wakako and Tanimura, Takahito},
  title        = {Soft‑Label Caching and Sharpening for Communication‑Efficient Federated Distillation},
  note         = {Manuscript under review},
  year         = {2025},
  url          = {https://github.com/kitsuyaazuma/SCARLET},
}
```
