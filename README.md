# Abismal_Torch
Testing PyTorch implementation of [absimal](https://github.com/rs-station/abismal)

![Build](https://github.com/rs-station/reciprocalspaceship/workflows/Build/badge.svg)
[![codecov](https://codecov.io/gh/DorisMai/abismal_torch/graph/badge.svg?token=VS8SANGY1B)](https://codecov.io/gh/DorisMai/abismal_torch)

## Installation
```bash
# make a new conda environment
conda create -n abismal_torch python=3.12
conda activate abismal_torch

# install torch
pip install torch
# ===== optional: test GPU support by torch =====
# python -c "import torch; print(torch.cuda.is_available())"

# install abismal-torch
pip install abismal-torch
# ===== alternatively, if you want the latest version from source or make changes to the code =====
# git clone git@github.com:DorisMai/abismal_torch.git
# cd abismal_torch
# pip install -e .
```

## Usage
Basic usage:

```bash
abismal-yaml fit
```
Example of overriding default values by command line flags:

```bash
abismal-yaml fit --data.dmin 1.5 --mtz_output.save_every_n_epoch=10
```

Example of overriding default values by yaml files:

```bash
abismal-yaml fit -c abismal_torch/command_line/configs/custom_config.yaml
```
