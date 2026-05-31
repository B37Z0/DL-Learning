import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tempfile import TemporaryDirectory
import matplotlib.pyplot as plt
import numpy as np
import os

# For GPU acceleration
# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
# print(f"Using {device} device")
# import torch.backends.cudnn as cudnn
# cudnn.benchmark = True


## Preprocessing ##
data_transforms = {
    # Training data - augmentation + normalization
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    # Validation data - normalization
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

img_datasets = {x: datasets.ImageFolder(os.path.join('data', x), data_transforms[x])
               for x in ['train', 'val']}
dataloaders = {x: DataLoader(img_datasets[x], batch_size=4, shuffle=True) # num_workers=4
              for x in ['train', 'val']}
dataset_sizes = {x: len(img_datasets[x]) 
                for x in ['train', 'val']}
class_names = img_datasets['train'].classes

# Helper function to display images
def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause so plots update

inputs, classes = next(iter(dataloaders['train'])) # get batch
out = torchvision.utils.make_grid(inputs) # make grid from batch
imshow(out, title=[class_names[x] for x in classes]) # show grid


## Train/Eval Function ##
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):

    # Use temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 25)

            for phase in ['train', 'val']:
                if phase == 'train': model.train()
                else: model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    # inputs, labels = inputs.to(device), labels.to(device)

                    # If training, enable grad tracking
                    # Block still runs during validation, just w/o grad tracking
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        # torch.max() returns a tuple of (values, indices)
                        # We only need the index (corresponding to the predicted class)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                    # Total loss per batch (multiply by batch size because loss is averaged)
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                
                # Step scheduler for learning rate decay
                if phase == 'train':
                    scheduler.step()

                # Avg loss and accuracy per epoch
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                
                # Save model if validation accuracy improves
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_params_path)

            print() # spacing
        print(f'Best val Acc: {best_acc:4f}')

        # Load best model
        model.load_state_dict(torch.load(best_params_path, weights_only=True))
    return model

# Helper function to visualize model predictions
def visualize_model(model, num_images=6):
    # Save current mode to restore
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            # inputs = inputs.to(device)
            # labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        plt.tight_layout()
        plt.show()
        model.train(mode=was_training)


## Fine-tuning *** ##
model_ft = models.resnet18(weights='IMAGENET1K_V1')
# Replace final layer
num_ftrs = model_ft.fc.in_features 
model_ft.fc = nn.Linear(num_ftrs, 2) # only have 2 classes
# model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()
# Observe that all parameters are being optimized ***
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)
torch.save(model_ft.state_dict(), 'resnet18_ft.pth')
print("Saved model_ft to resnet18_ft.pth")
visualize_model(model_ft)


## Fixed Feature Extraction *** ##
model_conv = models.resnet18(weights='IMAGENET1K_V1')
# Freeze all layers ***
for param in model_conv.parameters():
    param.requires_grad = False

# Replace final layer - requires grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)
# model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()
# Observe that only final layer parameters are being optimized ***
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=25)
torch.save(model_conv.state_dict(), 'resnet18_conv.pth')
print("Saved model_conv to resnet18_conv.pth")
visualize_model(model_conv)
