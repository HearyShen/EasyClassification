# EasyClassification

[**EasyClassification (easycls)**](https://github.com/HearyShen/EasyClassification) is a classification framework based on PyTorch.

## Features

A deep-learning classification framework with clean code and fine extensibility.

Main features: 

- CPU and GPU(CUDA) support
- YAML configuration file for all functions and parameters
- auto log system

## Install and Run

### Requirements

- python >= 3.6

- pytorch

- pyyaml

### Configurate

Configurate sections in `config.yml`:

- **basic**: basic train/eval settings
- **data**: dataset and path
- **model**: model architecture and parameters
- **loss**: loss function and parameters
- **optimizer**: optimizer on model parameters
- **lr_scheduler**: learning rate scheduler (usually learning rate decay)
- **speed**: computing speed related settings

### Train

Train a new model: 

```bash
python tools\train.py -c config.yml 
```

Resume training model with a previous checkpoint:

```bash
python tools\train.py -c config.yml -r <name>.pt
```

### Eval

Eval a model with a checkpoint:

```bash
python tools\eval.py -c config.yml -r <name>.pt
```

