import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
from torchvision import datasets, models, transforms
from torch.utils.data import random_split,WeightedRandomSampler
from collections import Counter
from PIL import Image

#Data augmentation
batch_size = 64
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p = 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ]),
    'test': transforms.Compose([
        transforms.Resize([32]),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
}
# Load CIFAR-10 dataset

image_datasets = {
'train': datasets.CIFAR10(root='./data', train=True, download=True, transform=data_transforms['train']),
'test': datasets.CIFAR10(root='./data', train=False, download=True, transform=data_transforms['test'])
}
#Setting training datasets to 40k
train, _ = random_split(image_datasets['train'], [40000, len(image_datasets['train']) - 40000])

# Access the full dataset's targets via train.dataset
targets = train.dataset.targets

# Extract targets for the subset
subset_targets = [targets[i] for i in train.indices]
subset_class_counts = Counter(subset_targets)

# Calculate weights for each class in the subset
subset_num_samples = len(subset_targets)
subset_class_weights = {cls: subset_num_samples / count for cls, count in subset_class_counts.items()}
subset_sample_weights = [subset_class_weights[cls] for cls in subset_targets]

# Create the sampler for the subset
sampler = WeightedRandomSampler(subset_sample_weights, num_samples=len(subset_sample_weights), replacement=True)


dataloaders= {
  'train': torch.utils.data.DataLoader(train, batch_size=64, sampler=sampler),
  'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=64, shuffle=False)
}

# Calculate dataset sizes and get class names
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['train'].classes


print(f"Training images count: {len(train)}")
print(f"Test images count: {dataset_sizes['test']}")
print(f"Class names:{class_names}")



# Loading of Resnet 34
model = models.resnet34(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 10)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

device = torch.device("cuda" if torch.cuda.is_available()else "cpu")
if torch.cuda.is_available():
  print('Cuda is using GPU')
else:
  print('Cuda is using CPU')

model = model.to(device)

# Training loop
for epoch in range(30):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in dataloaders['train']:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloaders['train'])
    epoch_acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{30}], Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.2f}%")

# Evaluation on Test Set
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in dataloaders['test']:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
print(f"Test Accuracy: {test_accuracy:.2f}%")
