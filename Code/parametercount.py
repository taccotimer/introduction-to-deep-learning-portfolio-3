import torch

state_dict = torch.load("ResNet18_cells_best.pth", map_location="cpu")


total = sum(t.numel() for t in state_dict.values())
print(f"{total:,} Parameter")