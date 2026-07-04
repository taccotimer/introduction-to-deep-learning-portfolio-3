# Consolidated Benchmark Report

## Summary Table

### CELLS (min. accuracy: 90.0%)

| Model    | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | Meets Expectation |
|----------|-------------:|--------------:|-----------:|-------------:|:------------------:|
| AlexNet  | 94.18 | 95.47 | 92.56 | 93.55 | ✓ Yes |
| VGG16    | 97.60 | 97.86 | 97.44 | 97.62 | ✓ Yes |
| ResNet18 | 97.87 | 98.02 | 97.86 | 97.93 | ✓ Yes |

### CHEST (min. accuracy: 87.0%)

| Model    | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | Meets Expectation |
|----------|-------------:|--------------:|-----------:|-------------:|:------------------:|
| AlexNet  | 87.34 | 91.33 | 83.21 | 85.29 | ✓ Yes |
| VGG16    | 88.78 | 92.39 | 85.04 | 87.09 | ✓ Yes |
| ResNet18 | 86.86 | 88.95 | 83.42 | 85.07 | ✗ No  |

### LESIONS (min. accuracy: 67.0%)

| Model    | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | Meets Expectation |
|----------|-------------:|--------------:|-----------:|-------------:|:------------------:|
| AlexNet  | 75.11 | 59.58 | 47.68 | 52.08 | ✓ Yes |
| VGG16    | 71.22 | 48.85 | 47.63 | 46.30 | ✓ Yes |
| ResNet18 | 75.61 | 60.25 | 53.06 | 54.86 | ✓ Yes |

### ORGANS (min. accuracy: 83.0%)

| Model    | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | Meets Expectation |
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
| Organs  | ResNet18 | Best across all 4 metrics |
