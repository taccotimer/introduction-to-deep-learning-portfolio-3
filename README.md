# IDL Final Project – Medical Image Classification

**Name(s):** Josian Müller, Tim Elflein
**Enrollment number(s):** 10013466 , 10013032

## What does this project do?

We train and test several neural networks (AlexNet, VGG16, ResNet18, and
smaller variant of AlexNet) on medical image datasets and compare their
accuracy, speed, and memory usage.

## Folder structure

```
Code/
  main.py                 Runs training + testing for all configured models
  train.py                Trains a single model (based on a config)
  fit.py                  Training loop (Trainer class)
  models.py               All model architectures
  data.py                 Loads the data and builds the DataLoaders
  training_framework.py   Runs training for several configs in a row
  testing_framework.py    Tests trained models, builds the results table
  parametercount.py       Gives the number of parameters of a model
config_files/              Two json files `config.json' and `testing_config.json' where models and datasets are configured
data/                       Datasets (.pt files, see below)
*_best.pth                  Saved, best weights after training for each model, the models with use_pretrained are models which used transfer leaning
AUDIT_LOG.md                List of found and fixed bugs
REPORT.md                   Results and evaluation
```

## Requirements

- Python 3.10 or newer
- pip
- Optional: NVIDIA GPU with CUDA (training also works on CPU, just slower, memory usage only available with CUDA)

## Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd introduction-to-deep-learning-portfolio-3

# 2. Create and activate a virtual environment (Windows)
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

By default, `pip install -r requirements.txt` installs the **CPU-only** build
of torch. If you have an NVIDIA GPU and want to train on it, install the
CUDA build of torch instead (example for CUDA 11.8):

```bash
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
```

## Download the data

Download the datasets here: https://cloud.fiw.fhws.de/s/LpYa2dCW85kwdNn

Place the downloaded `.pt` files in the `data/` folder.

## Start training

```bash
cd Code
python training_framework.py
```

Which models/datasets get trained is set in `training_framework.py` in the
`config_files` list. Each config file lives in `config_files/`.

## Testing

```bash
cd Code
python testing_framework.py
```

This loads the saved `*_best.pth` files and prints a results table
(accuracy, precision, recall, F1-score, inference time and peak memory usage). <br>
In fit.py in the fit method you can set the parameters for early stopping. For the experiments and basic Training that feature was disabled.

## Training + testing together

```bash
cd Code
python main.py
```

## Logging

Logs of training and testing are save in separate log files:
- fit.log / log of the training runs
- test.log / log of the testing runs

# Configuration Files
 
This repo uses two JSON config files: one for **training** and one for **testing**.
 
## `config.json` — training runs
 
An **array of run objects**. Each object is one training run, and every run in the list is executed in order. Fields:
 
| Key | Type | Description |
|-----|------|-------------|
| `DATA` | string | Dataset name (e.g. `cells`, `chest`, `lesions`, `orgs`, `organs`). |
| `DATA_PATH` | string | Root directory where datasets live (e.g. `./data`). |
| `BATCH_SIZE` | int | Mini-batch size. |
| `MODEL` | string | Architecture to train (`AlexNet`, `VGG16`, `ResNet18`). |
| `ACTIVATION` | string | Activation function (e.g. `ReLU`). |
| `LEARNING_RATE` | float | Optimizer learning rate. |
| `DROP_RATE` | float | Dropout probability. |
| `EPOCHS` | int | Number of training epochs. |
 
### Optional keys (transfer learning)
 
| Key | Type | Description |
|-----|------|-------------|
| `TRANSFER_LEARNING` | bool | If `true`, initialize from pretrained weights instead of training from scratch. |
| `PRETRAINED_MODEL_PATH` | string | Path to the checkpoint to load. **Required when `TRANSFER_LEARNING` is `true`.** |
 
### Example
 
```json
{
  "DATA": "organs",
  "DATA_PATH": "./data",
  "BATCH_SIZE": 64,
  "MODEL": "AlexNet",
  "ACTIVATION": "ReLU",
  "LEARNING_RATE": 0.0001,
  "DROP_RATE": 0.3,
  "EPOCHS": 50,
  "TRANSFER_LEARNING": true,
  "PRETRAINED_MODEL_PATH": "./AlexNet_orgs_best.pth"
}
```
 
## `testing_config.json` — evaluation
 
A **single object** describing which trained models to evaluate and on which datasets. Fields:
 
| Key | Type | Description |
|-----|------|-------------|
| `datapath` | string | Root directory for the test data. |
| `models_list` | string[] | Architectures to evaluate. |
| `datasets` | object[] | Datasets to test on. Each has a `name`, plus an optional `use_transfer_learning` flag to select the transfer-learned vs. from-scratch checkpoint. |
 
### Example
 
```json
{
  "datapath": "data/",
  "models_list": ["AlexNet", "VGG16", "ResNet18", "SlimAlexNet"],
  "datasets": [
    {"name": "cells"},
    {"name": "organs", "use_transfer_learning": true},
    {"name": "organs", "use_transfer_learning": false}
  ]
}
```
