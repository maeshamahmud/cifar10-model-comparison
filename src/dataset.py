import torch
import torchvision
from torch.utils.data import Subset
import torchvision.transforms as transforms

def load_cifar10():
    transform = transforms.ToTensor()

    # Load training set
    trainset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    # Load test set
    testset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    return trainset, testset

def select_n_per_class(dataset, n_per_class, num_classes=10):
    labels = dataset.targets
    selected_indices = []
    for c in range(num_classes):
        class_indices = [i for i, label in enumerate(labels) if label == c]
        class_indices = class_indices[:n_per_class]
        selected_indices.extend(class_indices)
    subset = Subset(dataset, selected_indices)
    return subset

def get_balanced_cifar10():
    transform = transforms.ToTensor()

    train_full = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    test_full = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    train_balanced = select_n_per_class(train_full, n_per_class=500)
    test_balanced = select_n_per_class(test_full, n_per_class=100)

    return train_balanced, test_balanced
