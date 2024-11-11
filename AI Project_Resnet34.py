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

train_dir = "C:/Users/Asus/Desktop/datasets/train_2000";
test_dir = "C:/Users/Asus/Desktop/datasets/test";
val_dir = "C:/Users/Asus/Desktop/datasets/val_500";
checkpointpath = 'checkpoint.tar'
submissionfile = 'submission.csv'

device = torch.device("cuda" if torch.cuda.is_available()else "cpu")
if torch.cuda.is_available():
  print('Cuda is using GPU')
else:
  print('Cuda is using CPU')
  
batch_size = 32
# Data Preprocessing Portion
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {
    'train': datasets.ImageFolder(train_dir, data_transforms['train']),
    'val': datasets.ImageFolder(val_dir, data_transforms['val'])
}

dataloaders= {
    'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=4, shuffle=True, num_workers=4),
    'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=4, shuffle=False, num_workers=4)}



# Calculate dataset sizes and get class names
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes


print(f'Training images count: {dataset_sizes["train"]}')
print(f'Validation images count: {dataset_sizes["val"]}')
print(f'Class names:{class_names}')

## Second Part of running the code
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp) 


inputs, classes = next(iter(dataloaders['train']))
sample_train_images = torchvision.utils.make_grid(inputs)
imshow(sample_train_images, title=classes)

def train_model(model, criterion, optimizer, num_epochs=30, checkpoint = None):
    since = time.time()
    
    if checkpoint is None:
        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = math.inf
        best_acc = 0.0
        
    else:
        print(f'Val loss: {checkpoint["best_val_loss"]}, Val accuracy: {checkpoint["best_val_accuracy"]}')
        model.load_state_dict(checkpoint['model_state_dict'])
        best_model_wts = copy.deepcopy(model.state_dict())
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_loss = checkpoint['best_val_loss']
        best_acc = checkpoint['best_val_accuracy']
        
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0
            
# Data iteration
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
# zeroing of parameter gradients
                optimizer.zero_grad() 
                if i % 200 == 199:
                    print('[%d, %d] loss: %.3f' % 
                          (epoch + 1, i, running_loss / (i * inputs.size(0))))
                    
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
# backward pass and optimization during training
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

# Deep copying of model
            if phase == 'val' and epoch_loss < best_loss:
                print(f'New best model found!')
                print(f'New record loss: {epoch_loss}, previous record loss: {best_loss}')
                best_loss = epoch_loss
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:.4f} Best val loss: {:.4f}'.format(best_acc, best_loss))
    
# load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_loss, best_acc

#Define resnet34 model
model_resnet34 = torch.hub.load('pytorch/vision','resnet34', pretrained = True)

#Freezing of parameters
for name, param in model_resnet34.named_parameters():
        param.requires_grad = False
        

num_ftrs = model_resnet34.fc.in_features
model_resnet34.fc = nn.Linear(num_ftrs,2)
model_resnet34 = model_resnet34.to(device)
criterion= nn.CrossEntropyLoss()
optimizer = optim.Adam(model_resnet34.parameters(), lr=0.001)

try:
    checkpoint = torch.load(checkpointpath)
    print("ckpt loaded")
except:
    checkpoint = None
    print("No chkpt")

model_resnet34, best_val_loss, best_val_acc = train_model(model_resnet34,
                                                      criterion,
                                                      optimizer, 
                                                      num_epochs=30,  
                                                      checkpoint=checkpoint)

# Saving of model and optimizer state
torch.save({'model_state_dict': model_resnet34.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'best_val_accuracy': best_val_acc,
            }, checkpointpath)

for param in model_resnet34.parameters():
    param.requires_grad = True
model_resnet34 = model_resnet34.to(device)

optimizer_ft = optim.Adam(model_resnet34.parameters(), lr=0.001)


#Model visualization
def visualize_model(model,images=4):
    was_training=model.training
    model.eval()
    images_so_far=0
    fig=plt.figure()
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs=inputs.to(device)
            labels=labels.to(device)
            outputs=model(inputs)
            _,preds=torch.max(outputs, 1)
            
            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
visualize_model(model_resnet34)
plt.ioff()
plt.show()
 
# Preprocess of test images for inference
def apply_test_transforms(inp):
    out = transforms.functional.resize(inp, [224,224])
    out = transforms.functional.to_tensor(out)
    out = transforms.functional.normalize(out, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return out



# Func to perform checking on how the model classifies the desired test image
test_files = os.listdir(test_dir)
im = Image.open(f"C:/Users/Asus/Desktop/datasets/test/358.jpg")
plt.imshow(im)
plt.show()
    
model_resnet34.to(device)
im_as_tensor = apply_test_transforms(im)
im_as_tensor = im_as_tensor.to(device)
print(im_as_tensor.size())
minibatch = torch.stack([im_as_tensor])
print(minibatch.size())

model_resnet34(minibatch)
softmax = nn.Softmax(dim=1)
preds=softmax(model_resnet34(minibatch))
print(f"Results:{preds}")

if preds[0][0]> preds[0][1]:
    print(f"This image is a cat")
else:
    print(f"This image is a dog")


def extract_fileid(fname):
    match = re.search(r'\d+', fname)  # Extract the first sequence of digits
    if match:
        return int(match.group())
    else:
        raise ValueError(f"No numeric ID found in filename: {fname}")


def imageclassification(model,tensor):
    model.eval()
    with torch.no_grad():
        output = model(tensor.unsqueeze(0).to(device)) 
        preds = nn.Softmax(dim=1)(output) 
        return torch.argmax(preds, dim=1).item() 
    
idLabel={}

for fname in test_files:
    try:
        im=Image.open(f'{test_dir}/{fname}')
        im_as_tensor=apply_test_transforms(im)
        label=imageclassification(model_resnet34, im_as_tensor)
        idLabel[extract_fileid(fname)]=label
    except Exception as e:
        print(f"Error processing file {fname}: {e}")
        
    
    
df=pd.DataFrame(list(idLabel.items()),columns=['id','Label'])
df.to_csv('submission.csv', index=False)
