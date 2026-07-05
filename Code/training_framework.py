import json

from train import main as train_main


def train_main_runs(
    config_file="config_files/config.json",

):
    
    with open(config_file, "r") as f:
        array_configs = json.load(f)
    for config in array_configs:
        train_main(config)


if __name__ == "__main__":
    train_main_runs()
