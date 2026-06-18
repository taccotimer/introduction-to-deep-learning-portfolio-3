import json

from train import main


def train_main_runs():
    for config_file in [
        "config_files/AlexNet_config.json",
        "config_files/ResNet18_config.json",
        "config_files/VGG16_config.json",
    ]:
        with open(config_file, "r") as f:
            array_configs = json.load(f)
        for config in array_configs:
            print(f"Training with config: {config_file}")
            main(config)


if __name__ == "__main__":
    train_main_runs()
