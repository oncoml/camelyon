# Training Camelyon17 images on ResNet with Pytorch
# From the pytorch.org and blog of Chris Fotache

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data.sampler import WeightedRandomSampler
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import ImageFile

train_dir = '/home/steveyang/projects/camelyon17/tile_images/resnet/cross_val_0/train'
val_dir = '/home/steveyang/projects/camelyon17/tile_images/resnet/cross_val_0/val'
train_mean = (0.6985210180282593, 0.3701254725456238, 0.656425416469574) 
train_std = (0.1374625712633133, 0.16258588433265686, 0.10656347870826721)
val_mean = (0.7268304228782654, 0.4075404107570648, 0.6833457946777344)
val_std = (0.13726474344730377, 0.16230422258377075, 0.1068471297621727)


def load_data(train_dir, val_dir):
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    train_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(), 
                                        transforms.ToTensor(),
                                        transforms.Normalize(train_mean, train_std)])
    val_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.ToTensor(),
                                        transforms.Normalize(val_mean, val_std)])

    train_data = datasets.ImageFolder(root=train_dir, transform=train_transforms)
    val_data = datasets.ImageFolder(root=val_dir, transform=val_transforms)
    
    train_weights = calculate_weight(train_data.imgs, len(train_data.classes))
    train_weights = torch.DoubleTensor(train_weights)
    val_weights = calculate_weight(val_data.imgs, len(val_data.classes))
    val_weights = torch.DoubleTensor(val_weights)

    train_sampler = WeightedRandomSampler(train_weights, len(train_weights))
    val_sampler = WeightedRandomSampler(val_weights, len(val_weights))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=16, sampler=val_sampler)

    return train_loader, val_loader



def train(train_dir, val_dir, checkpoint_file=None):

    train_loader, val_loader = load_data(train_dir, val_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.wide_resnet50_2(pretrained=False)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(nn.Linear(2048, 512), 
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(512, 1))

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adadelta(model.parameters(), lr=0.001)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.20, patience=1)
    model.to(device)

    if checkpoint_file is not None:
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        last_epoch = checkpoint['epoch']
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        val_accuracies = checkpoint['val_accuracies']
        train_accuracies = checkpoint['train_accuracies']
    else:
        last_epoch = 0
        train_losses, val_losses =  [], []
        train_accuracies, val_accuracies = [], []


    epochs = 10 
    steps = 0
    print_every = 500

    for epoch in range(last_epoch+1, last_epoch+epochs+1):
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model(inputs)
            labels = labels.unsqueeze(1).float()
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            model.eval()
            train_probability = torch.sigmoid(logps.data)
            train_predicted = (train_probability > 0.5).type(torch.cuda.FloatTensor)
            train_total += labels.size(0)
            train_correct += (train_predicted == labels).sum().item()
            train_accuracy = train_correct / train_total
            model.train()

            if steps % print_every == 0:
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                model.eval()

                with torch.no_grad():
                    for val_inputs, val_labels in val_loader:
                        val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                        val_logps = model(val_inputs)
                        val_labels = val_labels.unsqueeze(1).float()
                        batch_loss = criterion(val_logps, val_labels)
                        val_loss += batch_loss.item()
                        probability = torch.sigmoid(val_logps.data)
                        predicted = (probability > 0.5).type(torch.cuda.FloatTensor)
                        val_total += val_labels.size(0)
                        val_correct += (predicted == val_labels).sum().item()
                        val_accuracy = val_correct / val_total
                    #scheduler.step(val_loss / len(val_loader))

                train_losses.append(running_loss / print_every)
                val_losses.append(val_loss / len(val_loader))
                val_accuracies.append(val_accuracy)
                train_accuracies.append(train_accuracy)

                print(f"Epoch {epoch}/{last_epoch+epochs}.. "
                  f"Step {steps}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Val loss: {val_loss/len(val_loader):.3f}... "
                  f"Train accuracy: {train_accuracy:.3f}.. "
                  f"Val accuracy: {val_accuracy:.3f}")
                running_loss = 0.0
                model.train()
    
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,}

    torch.save(state, 'tumor_model_wideresnet50_lr0.001_adadelta_10epochs_fn.pth')


def calculate_weight(images, nclasses):
    """ Method to calculate weights for balance between classes.
        From Jordi de la Torre
    """
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for index, val in enumerate(images):
        weight[index] = weight_per_class[val[1]]
    return weight
