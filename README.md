# Adaptive Milling Model Training

Model training package for [Adaptive Milling](https://github.com/rosalindfranklininstitute/adaptive_milling).

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![CI](https://github.com/rosalindfranklininstitute/AM-model-training/actions/workflows/python-test.yaml/badge.svg)](https://github.com/rosalindfranklininstitute/AM-model-training/actions/workflows/python-test.yaml)

## Installation

This package is available from PyPI as [am-model-training](https://pypi.org/p/am-model-training) and can be installed via the following steps:

1. Decide where you want to create your Python virtual environment and open a terminal in that location.
2. Follow the one of the sets of instructions below. We recommend using 'uv' as it handles installing the appropriate torch backend automatically.

### [uv](https://docs.astral.sh/uv/getting-started/installation/) (recommended)

```Shell
# Create a virtual environment
uv venv

# Activate the virtual environment (Windows)
.venv\Scripts\activate

# Install including AutoLamella from fibsemOS with the appropriate PyTorch backend for your machine
uv pip install am-model-training --torch-backend auto
```

To activate the virtual environment on Linux systems: `source .venv/bin/activate`

### [miniforge](https://conda-forge.org/download/)

```Shell
# Create a virtual environment
conda create -p ./.venv python pip

# Activate the virtual environment
conda activate ./.venv

# Install PyTorch (select appropriate compute platform, see table below)
python -m pip install pytorch --index-url <Index URL>

# Install including AutoLamella from fibsemOS
python -m pip install -e am-model-training
```

If using CUDA, the version must be below or equal to the the system CUDA version, which can be checked with the command `nvidia-smi`.

| Compute Platform | Index URL                              |
| ---------------- | -------------------------------------- |
| CPU              | https://download.pytorch.org/whl/cpu   |
| CUDA 11.3        | https://download.pytorch.org/whl/cu113 |
| CUDA 11.8        | https://download.pytorch.org/whl/cu118 |
| CUDA 12.6        | https://download.pytorch.org/whl/cu126 |
| CUDA 12.8        | https://download.pytorch.org/whl/cu128 |

## Usage

### Train a new model: `am-train`

```
usage: am-train [-h] -p CSV_PATH -o OUTPUT_DIRECTORY [-v VALIDATION_PATH] [-s VALIDATION_SPLIT] [-e MAX_EPOCHS] [--frozen FROZEN_EPOCHS] [-tb TRAINING_BATCH_SIZE] [-vb VALIDATION_BATCH_SIZE] [--gpu GPU_NUMBER] [--cpu] [--save-all]

CLI for training new models for Adaptive Milling

options:
  -h, --help            show this help message and exit
  -p CSV_PATH, --csv CSV_PATH
                        Path to a csv file listing paths to image-segmentation pairs. If this argument is given multiple times the files will be combined.
  -o OUTPUT_DIRECTORY, --output OUTPUT_DIRECTORY
                        Path to directory where models will be saved. Directory will be created if it doesn't already exist.
  -v VALIDATION_PATH, --validation-csv VALIDATION_PATH
                        Path to a csv file listing paths to image-segmentation pairs to be used for validation. This will override the --validation-split argument if specified. If this argument is given multiple times the files will be combined.
  -s VALIDATION_SPLIT, --validation-split VALIDATION_SPLIT
                        The validation split to use. This will be ignored if --validation-csv is given. (default: 0.2)
  -e MAX_EPOCHS, --max-epochs MAX_EPOCHS
                        The maximum number of epochs to refine for. (default: 100)
  --frozen FROZEN_EPOCHS
                        The number of epochs before the encoder is unfrozen. (default: 25)
  -tb TRAINING_BATCH_SIZE, --training-batch TRAINING_BATCH_SIZE
                        The training batch size. Larger numbers will run faster, smaller will require less memory. (default: 6)
  -vb VALIDATION_BATCH_SIZE, --validation-batch VALIDATION_BATCH_SIZE
                        The validation batch size. Larger numbers will run faster, smaller will require less memory. (default: 1)
  --gpu GPU_NUMBER      Number of the GPU that will be used. (default: 0)
  --cpu                 Use the CPU only, ignoring the --gpu setting. This will be extremely slow, so is not recommended.
  --save-all            Save model weights from all epochs. Otherwise, just the models that improve on previous epochs.
```

### Refine an existing model: `am-refine`

```
usage: am-refine [-h] -p CSV_PATH -o OUTPUT_DIRECTORY -w WEIGHTS_PATH [-v VALIDATION_PATH] [-s VALIDATION_SPLIT] [-e MAX_EPOCHS] [-tb TRAINING_BATCH_SIZE] [-vb VALIDATION_BATCH_SIZE] [--gpu GPU_NUMBER] [--cpu] [--save-all]

CLI for refining models for Adaptive Milling

options:
  -h, --help            show this help message and exit
  -p CSV_PATH, --csv CSV_PATH
                        Path to a csv file listing paths to image-segmentation pairs. If this argument is given multiple times the files will be combined.
  -o OUTPUT_DIRECTORY, --output OUTPUT_DIRECTORY
                        Path to directory where models will be saved. Directory will be created if it doesn't already exist.
  -w WEIGHTS_PATH, --weights WEIGHTS_PATH
                        Path to the weights file that will be refined.
  -v VALIDATION_PATH, --validation-csv VALIDATION_PATH
                        Path to a csv file listing paths to image-segmentation pairs to be used for validation. This will override the --validation-split argument if specified. If this argument is given multiple times the files will be combined.
  -s VALIDATION_SPLIT, --validation-split VALIDATION_SPLIT
                        The validation split to use. This will be ignored if --validation-csv is given. (default: 0.2)
  -e MAX_EPOCHS, --max-epochs MAX_EPOCHS
                        The maximum number of epochs to refine for. (default: 50)
  -tb TRAINING_BATCH_SIZE, --training-batch TRAINING_BATCH_SIZE
                        The training batch size. Larger numbers will run faster, smaller will require less memory. (default: 6)
  -vb VALIDATION_BATCH_SIZE, --validation-batch VALIDATION_BATCH_SIZE
                        The validation batch size. Larger numbers will run faster, smaller will require less memory. (default: 1)
  --gpu GPU_NUMBER      Number of the GPU that will be used. (default: 0)
  --cpu                 Use the CPU only, ignoring the --gpu setting. This will be extremely slow, so is not recommended.
  --save-all            Save model weights from all epochs. Otherwise, just the models that improve on previous epochs.
```

### Run inference on images: `am-infer`

```
usage: am-infer [-h] -p CSV_PATH -o OUTPUT_DIRECTORY -w WEIGHTS_PATH [-b BATCH_SIZE] [--gpu GPU_NUMBER] [--cpu]

CLI for inferring images using an Adaptive Milling model

options:
  -h, --help            show this help message and exit
  -p CSV_PATH, --csv CSV_PATH
                        Path to a csv file listing paths to image-segmentation pairs. If this argument is given multiple times the files will be combined.
  -o OUTPUT_DIRECTORY, --output OUTPUT_DIRECTORY
                        Path to directory where models will be saved. Directory will be created if it doesn't already exist.
  -w WEIGHTS_PATH, --weights WEIGHTS_PATH
                        Path to the weights file that will be refined.
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        The batch size. Larger numbers will run faster, smaller will require less memory. (default: 1)
  --gpu GPU_NUMBER      Number of the GPU that will be used. (default: 0)
  --cpu                 Use the CPU only, ignoring the --gpu setting. This will be extremely slow, so is not recommended.
```

## Testing

You can run the package tests using pytest:

```Shell
uv run pytest
```

## Issues

Please use the [GitHub issue tracker](https://github.com/rosalindfranklininstitute/AM-model-training/issues) to submit bugs or request features.

## Contributions

If you would like to help contribute to project, please read our [contribution](CONTRIBUTING.md) guide and [code of conduct](CODE_OF_CONDUCT.md).

## License

Copyright Rosalind Franklin Institute, 2025.

Distributed under the terms of the Apache-2.0 license, Adaptive Milling Model Training is free and open source software.
