# SCARLET
Official implementation of SCARLET (Semi-supervised federated distillation with global CAching and Reduced soft-Label EnTropy)

# Getting Started

## pyenv + Poetry

```bash
pyenv install 3.12.5

git clone -b reproducibility https://github.com/kitsuyaazuma/SCARLET.git
cd SCARLET
poetry env use 3.12.5
poetry install

cd src
poetry run python main.py +algorithm=scarlet
```

## Docker

```bash
git clone -b reproducibility https://github.com/kitsuyaazuma/SCARLET.git
cd SCARLET
docker build -t scarlet .
docker run -it --rm --gpus=all --name scarlet scarlet:latest \
    bash -c "cd src && poetry run python main.py +algorithm=scarlet"
```

# Configuration

All hyperparameters live under `src/config/` and are managed with [Hydra](https://github.com/facebookresearch/hydra). You can override any setting on the command line:

```bash
poetry run python main.py \
    dir_alpha=0.3 \
    num_parallels=10 \
    +algorithm=scarlet \
    algorithm.era_exponent=1.0 \
    algorithm.cache_duration=50
```

You can swap `+algorithm=scarlet` for any supported algorithm (`dsfl`, `cfd`, `selectivefd`, `comet`) to reproduce baselines.

```bash
poetry run python main.py \
    +algorithm=dsfl \
    algorithm.era_temperature=0.1
```

# Disclaimer

We include simplified, fairness‑oriented implementations of DS‑FL, CFD, Selective‑FD, and COMET only to enable direct comparison and reproducibility. However:

- Most of these methods did not publish their source code or full hyperparameter details (see Appendix B in the paper).
- Our versions may differ in subtle implementation choices.
- For faithful reproduction, please consult each original paper and implement from scratch if needed.

Use these baselines as a rough guide—not as definitive "official" code.

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
