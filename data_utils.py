import os
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split
from pathlib import Path

# Ensure Subset class is allowed to be unpickled (needed in PyTorch >= 2.6)
import torch.utils.data.dataset
torch.serialization.add_safe_globals([torch.utils.data.dataset.Subset])

def generate_distributed_data(num_clients: int, alpha: float, save_dir: str) -> None:
    np.random.seed(42)
    torch.manual_seed(42)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    labels = np.array(dataset.targets)
    num_classes = len(np.unique(labels))

    class_distribution = np.random.dirichlet([alpha] * num_clients, num_classes)
    client_indices = [[] for _ in range(num_clients)]
    for class_idx in range(num_classes):
        class_indices = np.where(labels == class_idx)[0]
        np.random.shuffle(class_indices)
        proportions = class_distribution[class_idx]
        proportions = np.cumsum(proportions)
        proportions = proportions / proportions[-1]
        split_indices = (proportions * len(class_indices)).astype(int)[:-1]
        split_indices = np.split(class_indices, split_indices)
        for client_idx, indices in enumerate(split_indices):
            client_indices[client_idx].extend(indices)

    for cid in range(num_clients):
        client_data = Subset(dataset, client_indices[cid])
        torch.save(client_data, os.path.join(save_dir, f'client_{cid}.pt'))

def load_client_data(cid: int, data_dir: str, batch_size: int):
    path = os.path.join(data_dir, f'client_{cid}.pt')
    client_data = torch.load(path, weights_only=False)  # Allow loading full object
    train_size = int(0.8 * len(client_data))
    val_size = len(client_data) - train_size
    train_data, val_data = random_split(client_data, [train_size, val_size])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
