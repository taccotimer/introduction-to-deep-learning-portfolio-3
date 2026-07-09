"""
MAI/IDL SS26 - Final assignment.

MG 6/6/2026
"""

import json

import models
import torch
import torch.nn as nn
import torch.optim as optim
from fit import Trainer
import os

from data import get_loaders

TRAINED_MODELS_DIR = "trained_models"

def main(config, with_seed=True):
    if with_seed:
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
    os.makedirs(TRAINED_MODELS_DIR, exist_ok=True)
    
    if config is None:
        print("No config provided, loading default config.json")
        with open("config.json", "r") as f:
            config = json.load(f)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.xpu.is_available():
        device = torch.device("xpu")
    else:
        device = torch.device("cpu")
    print(f"Training executing on device: {device}")

    train_loader, val_loader, _, num_classes, channels = get_loaders(
        data=config["DATA"],
        data_path=config["DATA_PATH"],
        batch_size=config["BATCH_SIZE"],
    )

    model_class = getattr(models, config["MODEL"])
    model = model_class(
        in_channels=channels,
        num_classes=num_classes,
        drop_rate=config["DROP_RATE"],
        activation_str=config["ACTIVATION"],
    ).to(device)
    use_pretrained = config.get("TRANSFER_LEARNING", False)
    if use_pretrained:
        pretrained_path = config.get("PRETRAINED_MODEL_PATH", "")
        if pretrained_path:
            model.load_state_dict(torch.load(pretrained_path, map_location=device))
        else:
            print(
                "Warning: TRANSFER_LEARNING is True, but no PRETRAINED_MODEL_PATH was provided."
            )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])

    trainer = Trainer(model, criterion, optimizer, device)
    trainer.fit(train_loader, val_loader, epochs=config["EPOCHS"])
    save_path = os.path.join(
        TRAINED_MODELS_DIR,
        f"{config['MODEL']}_{config['DATA']}{('_use_pretrained' if use_pretrained else '')}_best.pth",
    )
    torch.save(
        trainer.best_model_state,
        save_path,
    )


if __name__ == "__main__":
    main(None)
