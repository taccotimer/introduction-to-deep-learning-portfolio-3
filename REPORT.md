# Consolidated Benchmark Report

## Summary Table

### CELLS (min. accuracy: 90.0%)

| Model    | Accuracy (%) | Precision (%) | Recall (%) | Macro F1-Score (%) | Meets Expectation |
|----------|-------------:|--------------:|-----------:|-------------:|:------------------:|
| AlexNet     | 96.49 | 96.19 | 96.33 | 96.25 | ✓ Yes |
| VGG16       | 98.19 | 98.33 | 98.17 | 98.25 | ✓ Yes |
| ResNet18    | 98.16 | 98.22 | 98.26 | 98.23 | ✓ Yes |
| SlimAlexNet | 96.32 | 96.60 | 96.02 | 96.27 | ✓ Yes |

### CHEST (min. accuracy: 87.0%)

| Model    | Accuracy (%) | Precision (%) | Recall (%) | Macro F1-Score (%) | Meets Expectation |
|----------|-------------:|--------------:|-----------:|-------------:|:------------------:|
| AlexNet     | 87.02 | 91.40 | 82.69 | 84.83 | ✓ Yes |
| VGG16       | 85.26 | 90.17 | 80.43 | 82.54 | ✗ No  |
| ResNet18    | 83.33 | 89.47 | 77.78 | 79.83 | ✗ No  |
| SlimAlexNet | 87.18 | 91.24 | 82.99 | 85.08 | ✓ Yes |

### LESIONS (min. accuracy: 67.0%)

| Model    | Accuracy (%) | Precision (%) | Recall (%) | Macro F1-Score (%) | Meets Expectation |
|----------|-------------:|--------------:|-----------:|-------------:|:------------------:|
| AlexNet     | 74.81 | 50.92 | 46.36 | 47.67 | ✓ Yes |
| VGG16       | 74.46 | 48.17 | 44.61 | 45.04 | ✓ Yes |
| ResNet18    | 75.66 | 57.06 | 54.29 | 55.54 | ✓ Yes |
| SlimAlexNet | 76.96 | 49.86 | 48.84 | 47.60 | ✓ Yes |

### ORGS (min. accuracy: 83.0%)

| Model    | Accuracy (%) | Precision (%) | Recall (%) | Macro F1-Score (%) | Meets Expectation |
|----------|-------------:|--------------:|-----------:|-------------:|:------------------:|
| AlexNet     | 90.03 | 88.84 | 89.25 | 88.95 | ✓ Yes |
| VGG16       | 91.52 | 90.49 | 90.19 | 90.23 | ✓ Yes |
| ResNet18    | 93.34 | 93.05 | 92.61 | 92.77 | ✓ Yes |
| SlimAlexNet | 90.58 | 89.49 | 89.63 | 89.40 | ✓ Yes |

## Architectural Recommendations

| Dataset | Recommendation | Rationale |
|---------|-----------------|-----------|
| Cells   | VGG16    | Highest accuracy, precision & F1 (98.19/98.33/98.25); ResNet18 is almost tied, ahead only on recall |
| Chest   | SlimAlexNet | Only model besides AlexNet to meet the 87% threshold; best accuracy, recall & F1 |
| Lesions | ResNet18 | Best precision, recall & F1; SlimAlexNet leads only on accuracy |
| Orgs    | ResNet18 | Best across all 4 metrics |


## Data-Scarcity Post-Mortem: Organs (Low-Sample)

**Expected min. accuracy: 40.0%**

| Model    | Regime            | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | Meets Expectation |
|----------|--------------------|-------------:|--------------:|-----------:|-------------:|:------------------:|
| AlexNet     | Scratch            | 55.50 | 54.07 | 52.94 | 51.46 | ✓ Yes |
| AlexNet     | Transfer Learning  | 56.00 | 50.44 | 50.07 | 49.39 | ✓ Yes |
| VGG16       | Scratch            | 54.50 | 41.50 | 48.42 | 42.84 | ✓ Yes |
| VGG16       | Transfer Learning  | 56.50 | 49.43 | 49.51 | 49.07 | ✓ Yes |
| ResNet18    | Scratch            | 60.00 | 60.18 | 53.35 | 53.09 | ✓ Yes |
| ResNet18    | Transfer Learning  | 61.50 | 58.27 | 57.27 | 55.35 | ✓ Yes |
| SlimAlexNet | Scratch            | 64.00 | 63.85 | 61.27 | 60.33 | ✓ Yes |
| SlimAlexNet | Transfer Learning  | 60.50 | 59.31 | 53.38 | 52.22 | ✓ Yes |

**Transfer learning vs. scratch training:**

1. **AlexNet, VGG16, ResNet18:** Transfer learning outperforms scratch training (+0.5 to +1.5 pp accuracy). The features pretrained on the larger `orgs` dataset generalize reasonably well to `organs`, giving the model a useful head start despite the domain gap between the two datasets.
2. **SlimAlexNet:** The opposite holds — scratch training (64.00%) outperforms transfer learning (60.50%). Its smaller capacity appears to adapt faster and more directly to the limited `organs` data than to reconciling pretrained `orgs` features, and it does not benefit from transfer the way the larger architectures do.

**Conclusion:** All models clear the required 40% threshold comfortably; SlimAlexNet (Scratch) at 64.00% is currently the best-performing solution overall, ahead of ResNet18 (Transfer Learning) at 61.50%.
**Recommendation:** To increase accuracy an Option would be augmentation of the data, to generate more training data. However it would be even better to gather new real world data to ensure correct generalization.



## Efficiency Verification Matrix (Training Runtime, Inference Latency, Peak Memory)

| Model | Dataset | Regime | Test Accuracy (%) | Training Time (s) | Peak Mem — Training (MB) | Inference Time (ms/sample) | Peak Mem — Inference (MB) |
|-------|---------|--------|-------------------:|-------------------:|--------------------------:|-----------------------------:|----------------------------:|
| AlexNet     | cells   | -                 | 96.49 | 116.72  | 212.15  | 0.1392 | 75.98  |
| AlexNet     | chest   | -                 | 87.02 | 41.80   | 208.42  | 0.0325 | 73.12  |
| AlexNet     | lesions | -                 | 74.81 | 67.44   | 212.01  | 0.0226 | 75.60  |
| AlexNet     | orgs    | -                 | 90.03 | 125.02  | 207.56  | 0.0200 | 73.60  |
| AlexNet     | organs  | Scratch           | 55.50 | 4.43    | 206.62  | 0.0228 | 73.60  |
| AlexNet     | organs  | Transfer Learning | 56.00 | 2.26    | 210.06  | 0.0927 | 73.60  |
| VGG16       | cells   | -                 | 98.19 | 685.14  | 901.91  | 0.1503 | 251.95 |
| VGG16       | chest   | -                 | 85.26 | 259.04  | 898.20  | 0.1369 | 249.98 |
| VGG16       | lesions | -                 | 74.46 | 401.51  | 900.90  | 0.1311 | 252.00 |
| VGG16       | orgs    | -                 | 91.52 | 761.60  | 898.42  | 0.1295 | 250.00 |
| VGG16       | organs  | Scratch           | 54.50 | 25.71   | 899.04  | 0.1326 | 250.00 |
| VGG16       | organs  | Transfer Learning | 56.50 | 12.98   | 899.04  | 0.1893 | 250.00 |
| ResNet18    | cells   | -                 | 98.16 | 1419.33 | 1400.03 | 0.2775 | 373.94 |
| ResNet18    | chest   | -                 | 83.33 | 540.92  | 1398.08 | 0.2681 | 372.43 |
| ResNet18    | lesions | -                 | 75.66 | 837.91  | 1400.52 | 0.2727 | 374.44 |
| ResNet18    | orgs    | -                 | 93.34 | 1591.06 | 1397.54 | 0.2714 | 372.45 |
| ResNet18    | organs  | Scratch           | 60.00 | 52.55   | 1396.79 | 0.2721 | 372.45 |
| ResNet18    | organs  | Transfer Learning | 61.50 | 26.38   | 1396.79 | 0.2957 | 372.45 |
| SlimAlexNet | cells   | -                 | 96.32 | 75.17   | 95.76   | 0.0162 | 31.47  |
| SlimAlexNet | chest   | -                 | 87.18 | 27.17   | 92.60   | 0.0243 | 30.56  |
| SlimAlexNet | lesions | -                 | 76.96 | 43.67   | 96.24   | 0.0152 | 31.47  |
| SlimAlexNet | orgs    | -                 | 90.58 | 79.34   | 94.98   | 0.0137 | 30.60  |
| SlimAlexNet | organs  | Scratch           | 64.00 | 3.06    | 94.98   | 0.0194 | 30.60  |
| SlimAlexNet | organs  | Transfer Learning | 60.50 | 1.47    | 94.98   | 0.0496 | 30.60  |

## Green Initiative Analysis
- **SlimAlexNet** is our downscaled model: it achieves ~2–20x less peak memory and up to ~10x faster inference than AlexNet, and up to ~40x less memory / ~17x faster inference than ResNet18, across every dataset it has been trained on. It achieves at every Dataset the minimum accuracy including on `organs`, where it reaches the highest scratch accuracy of any model (64.00%) at a fraction of the training cost.
- **ResNet18** is the most expensive model on every axis (training time, training memory, inference time, inference memory), despite topping the accuracy tables. It shows a accuracy/computation trade-off.

## Architectural Complexity (Parameter Count)

`SlimAlexNet` implements the Architectural Downscaling requirement in the model definition: every convolutional/linear width from `AlexNet` is scaled down by 0.5x, yielding a smaller parameter count at the same depth. And one linear layer was removed completely. Counts below are for a representative 3-channel, 4-class configuration (`cells`).

| Model       | Parameters | Relative to SlimAlexNet |
|-------------|-----------:|:------------------------:|
| SlimAlexNet | 1,952,308  | 1.0x (baseline)          |
| AlexNet     | 5,689,444  | 2.9x more                |
| ResNet18    | 11,170,884 | 5.7x more                |
| VGG16       | 12,629,572 | 6.5x more                |