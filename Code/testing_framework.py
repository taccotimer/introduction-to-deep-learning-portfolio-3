import logging
import time
from collections import defaultdict

import models
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from data import get_loaders

"""
Test all models on all datasets and compute comprehensive metrics.
Saves results to a summary table with accuracy, precision, recall, and F1-score.
"""


def test_models(
    datapath="./data",
    datasets=[
        {"name": "cells"},
        {"name": "chest"},
        {"name": "lesions"},
        {"name": "orgs"},
        {"name": "organs", "use_transfer_learning": True},
        {"name": "organs", "use_transfer_learning": False},
    ],
    models_list=[ "AlexNet", "VGG16", "ResNet18", "SlimAlexNet"],
):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler('test.log')
    console_handler = logging.StreamHandler()

    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    results = defaultdict(lambda: defaultdict(dict))

    for dataset in datasets:
        dataset_name = dataset["name"]
        use_transfer_learning = dataset.get("use_transfer_learning", False)
        _, _, test_loader, num_classes, channels = get_loaders(
            data=dataset_name, data_path=datapath, batch_size=64
        )

        for model_name in models_list:
            model_class = getattr(models, model_name)
            test_model = model_class(
                in_channels=channels,
                num_classes=num_classes,
                drop_rate=0.0,
                activation_str="ReLU",
            )

            if use_transfer_learning:
                name = f"{model_name}_{dataset_name}_use_pretrained_best.pth"
            else:
                name = f"{model_name}_{dataset_name}_best.pth"

            try:
                state_dict = torch.load(
                    name,
                    map_location="cpu",
                )
            except FileNotFoundError:
                logger.info(f"File not found: {name}")
                continue

            test_model.load_state_dict(state_dict)
            test_model.eval()

            all_predictions, all_labels = [], []
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            test_model.to(device)

            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)

            total_inference_time = 0.0
            total_samples = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    #synchronize and start timer
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    start_time = time.perf_counter()

                    outputs = test_model(images)
                    
                    #synchronize and stop timer
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    end_time = time.perf_counter()

                    # record time and batch size
                    batch_time = end_time - start_time
                    total_inference_time += batch_time
                    total_samples += images.size(0)

                    _, predicted = outputs.max(1)
                    all_predictions.extend(predicted.cpu().tolist())
                    all_labels.extend(labels.cpu().tolist())

            if device.type == 'cuda':
                peak_bytes = torch.cuda.max_memory_allocated(device)
                peak_mem_mb = peak_bytes / (1024 ** 2)
                mem_type = "GPU (VRAM)"

            avg_time_per_sample_sec = total_inference_time / total_samples
            avg_time_per_sample_ms = avg_time_per_sample_sec * 1000

            logger.info(f"{model_name}: inference time per sample: {avg_time_per_sample_ms:.4f} ms")
            if device.type == 'cuda':
                logger.info(f"Peak memory consumption: {peak_mem_mb:.2f} MB [{mem_type}]")

            all_predictions = np.array(all_predictions)
            all_labels = np.array(all_labels)

            accuracy = accuracy_score(all_labels, all_predictions)
            precision = precision_score(
                all_labels, all_predictions, average="macro", zero_division=0
            )
            recall = recall_score(
                all_labels, all_predictions, average="macro", zero_division=0
            )
            f1 = f1_score(all_labels, all_predictions, average="macro", zero_division=0)

            results[dataset_name][model_name][
                "use_transfer_learning" if use_transfer_learning else ""
            ] = {
                "accuracy": accuracy * 100,
                "precision": precision * 100,
                "recall": recall * 100,
                "f1_score": f1 * 100,
            }

    logger.info("\n" + "=" * 110)
    logger.info("SUMMARY TABLE: Model Performance Across All Datasets")
    logger.info("=" * 110)

    expected_min_acc = {
        "cells": 90.0,
        "chest": 87.0,
        "lesions": 67.0,
        "orgs": 83.0,
        "organs": 40.0,
    }

    for dataset in datasets:
        dataset_name = dataset["name"]
        use_transfer_learning = dataset.get("use_transfer_learning", False)
        logger.info(
            f"\n{dataset_name.upper()} {'with Transfer Learning' if use_transfer_learning else ''} (Expected min accuracy: {expected_min_acc[dataset_name]}%):"
        )
        logger.info(
            "| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | Meets Expectation |"
        )
        logger.info(f"|{'-'*12}|{'-'*15}|{'-'*16}|{'-'*13}|{'-'*15}|{'-'*20}|")

        for model_name in models_list:
            metrics = results[dataset_name][model_name][
                "use_transfer_learning" if use_transfer_learning else ""
            ]
            meets_exp = (
                "Yes"
                if metrics["accuracy"] >= expected_min_acc[dataset_name]
                else "No"
            )
            logger.info(
                f"| {model_name} | {metrics['accuracy']:.2f} | {metrics['precision']:.2f} | {metrics['recall']:.2f} | {metrics['f1_score']:.2f} | {meets_exp} |"
            )
    for handler in logger.handlers[:]:
        handler.close()               
        logger.removeHandler(handler)

    print("Logger successfully closed and file released.")


if __name__ == "__main__":
    test_models()
