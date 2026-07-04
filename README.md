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
config_files/              One JSON file per model/dataset (hyperparameters)
data/                       Datasets (.pt files, see below)
*_best.pth                  Saved, best weights after training for each model
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
(accuracy, precision, recall, F1-score).

## Training + testing together

```bash
cd Code
python main.py
```
