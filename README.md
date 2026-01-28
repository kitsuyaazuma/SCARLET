# SCARLET

Official implementation of SCARLET: "[Soft-Label Caching and Sharpening for Communication-Efficient Federated Distillation](https://ieeexplore.ieee.org/document/11344746)" (Accepted by IEEE TMC).

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
    bash -c "cd SCARLET/src && poetry run python main.py +algorithm=scarlet"
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

If you use this code in your research, please cite our paper:

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
