from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from data import get_loaders


def test_models():
    """
    Test all models on all datasets and compute comprehensive metrics.
    Saves results to a summary table with accuracy, precision, recall, and F1-score.
    """
    datapath = "./data"
    datasets = ["cells", "chest", "lesions", "orgs"]
    models = ["AlexNet", "VGG16", "ResNet18"]

    results = defaultdict(dict)

    for dataset in datasets:
        _, _, test_loader, num_classes, channels = get_loaders(
            data=dataset, data_path=datapath, batch_size=64
        )

        for model_name in models:
            test_model = torch.load(
                f"{model_name}_{dataset}_best.pth", weights_only=False
            )
            test_model.eval()

            all_predictions, all_labels = [], []
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            test_model.to(device)

            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = test_model(images)
                    _, predicted = outputs.max(1)
                    all_predictions.extend(predicted.cpu().tolist())
                    all_labels.extend(labels.cpu().tolist())

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

            results[dataset][model_name] = {
                "accuracy": accuracy * 100,
                "precision": precision * 100,
                "recall": recall * 100,
                "f1_score": f1 * 100,
            }
            # Print Summary Table

    print("\n" + "=" * 110)
    print("SUMMARY TABLE: Model Performance Across All Datasets")
    print("=" * 110)

    expected_min_acc = {"cells": 90.0, "chest": 87.0, "lesions": 67.0, "orgs": 83.0}

    for dataset in datasets:
        print(
            f"\n{dataset.upper()} (Expected min accuracy: {expected_min_acc[dataset]}%):"
        )
        print(
            "| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | Meets Expectation |"
        )
        print(f"| {'-'*12}'|{'-'*15}'|{'-'*16}'|{'-'*13}'|{'-'*15}'|{'-'*20}'|")

        for model_name in models:
            metrics = results[dataset][model_name]
            meets_exp = (
                "✓ Yes" if metrics["accuracy"] >= expected_min_acc[dataset] else "✗ No"
            )
            print(
                f"| {model_name} | {metrics['accuracy']:.2f} | {metrics['precision']:.2f} | {metrics['recall']:.2f} | {metrics['f1_score']:.2f} | {meets_exp} |"
            )


if __name__ == "__main__":
    test_models()
