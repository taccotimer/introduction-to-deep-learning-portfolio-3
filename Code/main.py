from testing_framework import test_models
from training_framework import train_main_runs

if __name__ == "__main__":
    train_main_runs(
        config_files=[
            "config_files/AlexNet_config.json",
            "config_files/ResNet18_config.json",
            "config_files/VGG16_config.json",
            "Transfer_Learning_Organs_config.json",
            "Organs_config.json"
        ]
    )

    test_models(
        datapath="./data",
        datasets=[
        {"name": "cells"},
        {"name": "chest"},
        {"name": "lesions"},
        {"name": "orgs"},
        {"name": "organs", "use_transfer_learning": True},
        {"name": "organs", "use_transfer_learning": False},
        ],
        models_list=["AlexNet", "VGG16", "ResNet18"],
    )
