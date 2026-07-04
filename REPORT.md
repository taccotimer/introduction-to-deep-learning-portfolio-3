# Consolidated Benchmark Report

## Summary Table

### CELLS (min. accuracy: 90.0%)

| Model    | Accuracy (%) | Precision (%) | Recall (%) | Macro F1-Score (%) | Meets Expectation |
|----------|-------------:|--------------:|-----------:|-------------:|:------------------:|
| AlexNet  | 94.18 | 95.47 | 92.56 | 93.55 | ✓ Yes |
| VGG16    | 97.60 | 97.86 | 97.44 | 97.62 | ✓ Yes |
| ResNet18 | 97.87 | 98.02 | 97.86 | 97.93 | ✓ Yes |

### CHEST (min. accuracy: 87.0%)

| Model    | Accuracy (%) | Precision (%) | Recall (%) | Macro F1-Score (%) | Meets Expectation |
|----------|-------------:|--------------:|-----------:|-------------:|:------------------:|
| AlexNet  | 87.34 | 91.33 | 83.21 | 85.29 | ✓ Yes |
| VGG16    | 88.78 | 92.39 | 85.04 | 87.09 | ✓ Yes |
| ResNet18 | 86.86 | 88.95 | 83.42 | 85.07 | ✗ No  |

### LESIONS (min. accuracy: 67.0%)

| Model    | Accuracy (%) | Precision (%) | Recall (%) | Macro F1-Score (%) | Meets Expectation |
|----------|-------------:|--------------:|-----------:|-------------:|:------------------:|
| AlexNet  | 75.11 | 59.58 | 47.68 | 52.08 | ✓ Yes |
| VGG16    | 71.22 | 48.85 | 47.63 | 46.30 | ✓ Yes |
| ResNet18 | 75.61 | 60.25 | 53.06 | 54.86 | ✓ Yes |

### ORGS (min. accuracy: 83.0%)

| Model    | Accuracy (%) | Precision (%) | Recall (%) | Macro F1-Score (%) | Meets Expectation |
|----------|-------------:|--------------:|-----------:|-------------:|:------------------:|
| AlexNet  | 90.15 | 89.54 | 88.33 | 88.77 | ✓ Yes |
| VGG16    | 90.54 | 89.74 | 89.53 | 89.53 | ✓ Yes |
| ResNet18 | 93.63 | 93.02 | 92.78 | 92.84 | ✓ Yes |

## Architectural Recommendations

| Dataset | Recommendation | Rationale |
|---------|-----------------|-----------|
| Cells   | ResNet18 | Best across all 4 metrics |
| Chest   | VGG16    | Best across all 4 metrics |
| Lesions | ResNet18 | Best across all 4 metrics |
| Orgs  | ResNet18 | Best across all 4 metrics |


## Data-Scarcity Post-Mortem: Organs (Low-Sample)

**Expected min. accuracy: 40.0%**

| Model    | Regime            | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | Meets Expectation |
|----------|--------------------|-------------:|--------------:|-----------:|-------------:|:------------------:|
| AlexNet  | Scratch            | 62.00 | 60.55 | 60.86 | 59.59 | ✓ Yes |
| AlexNet  | Transfer Learning  | 55.50 | 54.33 | 49.53 | 48.93 | ✓ Yes |
| VGG16    | Scratch            | 61.00 | 61.27 | 58.89 | 57.53 | ✓ Yes |
| VGG16    | Transfer Learning  | 57.50 | 50.96 | 49.20 | 49.00 | ✓ Yes |
| ResNet18 | Scratch            | 67.00 | 63.66 | 62.44 | 61.47 | ✓ Yes |
| ResNet18 | Transfer Learning  | 60.00 | 56.64 | 52.83 | 52.44 | ✓ Yes |

**Why transfer learning underperforms scratch training here:**

1. **Extremely small dataset:** With only 500 training images across 11 classes, there is likely too little data to properly adapt a large pretrained network without either underfitting the new classes or overwriting the pretrained features entirely.
2. **Domain gap:** The pretrained features don't align well with the new Organs data distribution (different class/image statistics), causing negative transfer. Hyperparameters (50 epochs, learning rate 0.001) were reasonable and consistent with the scratch runs, so the gap is not attributable to a training-configuration issue.

**Conclusion:** All models/regimes clear the required 40% threshold comfortably; ResNet18 (Scratch) at 67.00% is currently the best-performing solution.
