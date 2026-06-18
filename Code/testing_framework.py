import torch

from data import get_loaders


def test_models():
    datapath = "./data"
    datasets = ["cells", "chest", "lesions", "orgs"]
    models = ["AlexNet", "VGG16", "ResNet18"]
    for dataset in datasets:
        _, _, test_loader, num_classes, channels = get_loaders(
            data=dataset,
            data_path=datapath,
            batch_size=64,
        )
        for model in models:
            test_model = torch.load(f"{model}_{dataset}_best.pth")
            test_model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    outputs = test_model(images)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels.squeeze()).sum().item()


if __name__ == "__main__":
    test_models()
