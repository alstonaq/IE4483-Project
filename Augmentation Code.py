import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import time
import re
import os
import copy
import math
import pandas as pd
from PIL import Image

# Directories
train_augment_dir = "C:/Users/Asus/Desktop/datasets/train_500"
augmented_dir = "C:/Users/Asus/Desktop/datasets/augmented_500"

# Data Augmentation for `train_500`
os.makedirs(train_augment_dir, exist_ok=True)

augment_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation()
])

#Apply augmentations and save images
augment_dataset = datasets.ImageFolder(train_augment_dir, transform=augment_transform['train_500'])
for idx, (img, label) in enumerate(augment_dataset):
    label_name = augment_dataset.classes[label]  # "cat" or "dog"
    label_dir = os.path.join(train_augment_dir, label_name)
    os.makedirs(label_dir, exist_ok=True)
    img.save(os.path.join(label_dir, f"aug_{idx}.jpg"))  # Save augmented image

#Move images from train_500 to augmented_500
for label_name in augment_dataset.classes:
    src_dir = os.path.join(train_augment_dir, label_name)
    dest_dir = os.path.join(augmented_dir, label_name)
    os.makedirs(dest_dir, exist_ok=True)
    for filename in os.listdir(src_dir):
        os.rename(os.path.join(src_dir, filename), os.path.join(dest_dir, filename))