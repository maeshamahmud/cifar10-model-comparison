from dataset import load_cifar10
from dataset import get_balanced_cifar10
from collections import Counter

train_balanced, test_balanced = get_balanced_cifar10()
print(len(train_balanced))  # should be 5000
print(len(test_balanced))   # should be 1000
