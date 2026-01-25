# SCARLET
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fkitsuyaazuma%2FSCARLET%2Frefs%2Fheads%2Fmain%2Fpyproject.toml)
![GitHub License](https://img.shields.io/github/license/kitsuyaazuma/SCARLET)


Official implementation of **SCARLET** from the paper "[Soft-Label Caching and Sharpening for Communication-Efficient Federated Distillation](https://arxiv.org/abs/2504.19602)".

> [!IMPORTANT]
> The `main` branch contains a simplified implementation for better understanding of SCARLET’s core algorithms.
> For the exact experiment code and hyperparameter settings used in our paper, switch to the [`reproducibility`](https://github.com/kitsuyaazuma/SCARLET/tree/reproducibility) branch.

# Getting Started

## uv

```bash
git clone https://github.com/kitsuyaazuma/SCARLET.git
cd SCARLET
uv sync

uv run python main.py scarlet
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
uv run python main.py --help
```

# Citation

If you use this code in your research, please cite our preprint:

```bibtex
@misc{azuma2025softlabelcachingsharpeningcommunicationefficient,
      title={Soft-Label Caching and Sharpening for Communication-Efficient Federated Distillation}, 
      author={Kitsuya Azuma and Takayuki Nishio and Yuichi Kitagawa and Wakako Nakano and Takahito Tanimura},
      year={2025},
      eprint={2504.19602},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2504.19602}, 
}
```
