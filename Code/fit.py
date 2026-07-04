"""
MAI/IDL SS26 - Final assignment.

MG 6/6/2026
"""

import copy

import torch
import time

class Trainer:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.best_model_state = None

    def train_one_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0
        correct, sum = 0, 0

        for images, labels in dataloader:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)
            loss = self.criterion(outputs, labels.squeeze())

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            sum += labels.size(0)
            correct += predicted.eq(labels.squeeze()).sum().item()

        return running_loss / sum, (correct / sum) * 100

    def evaluate(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        correct, total = 0, 0

        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels.squeeze())

                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels.squeeze()).sum().item()

        return running_loss / total, (correct / total) * 100

    def fit(self, train_loader, val_loader, epochs, patience = 100, delta_improvement = 0.005):
        print("\n Starting Training Routine...")
        print("-" * 50)
        best_val_loss = float('inf')
        last_improvement = 0
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        start_training_time = time.perf_counter()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        for epoch in range(epochs):
            train_loss, train_acc = self.train_one_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)

            print(
                f"Epoch [{epoch+1:02d}/{epochs:02d}] | "
                f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%"
            )
            if val_loss < (best_val_loss - delta_improvement):
                best_val_loss = val_loss
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                last_improvement = 0
            else:
                last_improvement +=1
            if last_improvement > patience:
                print("-" * 50)
                print("Training Complete! - Early Stopping")
                break

        print("-" * 50)
        print("Training Complete!")
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        end_training_time = time.perf_counter()
        print(f"Total Training Time: {end_training_time - start_training_time:.2f} seconds")
        if self.device.type == "cuda":
            peak_memory_mb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
            print(f"Peak GPU Memory (allocated): {peak_memory_mb:.2f} MB")
