# Analyze performance of model

import torch
from torch import nn
import numpy as np
import sklearn.metrics as metrics
from train import load_data 
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import WeightedRandomSampler
from train import calculate_weight
from PIL import ImageFile
import pickle
import re
import statistics

checkpoint_path = 'tumor_model_wideresnet50_lr0.001_adadelta_10epochs_fn.pth'
data_dir = '/home/steveyang/projects/camelyon17/tile_images/resnet/cross_val_0/val'
val_mean = (0.7268304228782654, 0.4075404107570648, 0.6833457946777344)
val_std = (0.13726474344730377, 0.16230422258377075, 0.1068471297621727)

def analyze(model, data_loader):
    model.eval()

    groundtruth = torch.Tensor()
    prediction = torch.Tensor()
    probabilities = torch.Tensor()

    for val_inputs, val_labels in data_loader:
        val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
        val_logps = model(val_inputs)
        val_labels = val_labels.unsqueeze(1).float()
        groundtruth = torch.cat([groundtruth, val_labels], dim=0)
        probability = torch.sigmoid(val_logps.data)
        probabilities = torch.cat([probabilities, probability], dim=0)
        temp = (probability > 0.5).type(torch.FloatTensor)
        prediction = torch.cat([prediction, temp], dim=0)

    groundtruth_new = np.squeeze(groundtruth.numpy())
    probabilities_new = np.squeeze(probabilities.numpy())
    prediction_new = np.squeeze(prediction.numpy())

    return groundtruth_new, probabilities_new, prediction_new


def tp_fn_tiles(model, data_loader_with_paths):
    model.eval()

    tp_tiles = []
    fp_tiles = []
    fn_tiles = []
    tn_tiles = []

    for inputs, labels, paths in data_loader_with_paths:
        labels = labels.unsqueeze(1).float()
        outputs = model(inputs)
        probs = torch.sigmoid(outputs.data)
        preds = (probs > 0.5).type(torch.FloatTensor)

        for i in range(len(labels)):
            # True positives
            if labels[i] == 1 and preds[i] == 1:
                tp_tiles.append(paths[i])
            # False negatives
            elif labels[i] == 1 and preds[i] == 0:
                fn_tiles.append(paths[i])
            # False positives
            elif labels[i] == 0 and preds[i] == 1:
                fp_tiles.append(paths[i])
            elif labels[i] == 0 and preds[i] == 0:
                tn_tiles.append(paths[i])
            else:
                print("An error occurred when dividing into tp, fp, fn and tn")
                continue
    
    return tp_tiles, fp_tiles, fn_tiles, tn_tiles


def load_model(checkpoint_path, model_name):
    device = torch.device("cpu")

    if model_name == 'wide_resnet50':
        model = models.wide_resnet50_2(pretrained=False)
    else:
        print("Model not supported")

    model.fc = nn.Sequential(nn.Linear(2048, 512),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(512, 1))
    model.to(device)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


class ImageFolderWithPaths(datasets.ImageFolder):
    # From Andrew Jong

    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def load_data_with_paths(data_dir):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    data_transform = transforms.Compose([transforms.Resize(256),
                                        transforms.ToTensor(),
                                        transforms.Normalize(val_mean, val_std)])
    dataset = ImageFolderWithPaths(root=data_dir, transform=data_transform)
    weights = calculate_weight(dataset.imgs, len(dataset.classes))
    weights = torch.DoubleTensor(weights)
    sampler = WeightedRandomSampler(weights, len(weights))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=16, sampler=sampler)
    return data_loader


def calculate_area_mean_std(filepath_list, annotations_path):
    area_values = []

    with open(annotations_path, 'rb') as f:
        annotations = pickle.load(f)

    annotations = [item for sublist in annotations for item in sublist]

    for filepath in filepath_list:
        text = re.search('patient(.+?)png', filepath).group(0)
        
        for i in range(len(annotations)):
            tile_annot = annotations[i]
            if text == tile_annot[0] + '.png':
                if isinstance(tile_annot[1][0], list):
                    for i in range(len(tile_annot[1])):
                        area_values.append(calculate_area(tile_annot[1][i], tile_annot[2][i]))
                else:
                    area_values.append(calculate_area(tile_annot[1], tile_annot[2]))
    print("Area mean: ", statistics.mean(area_values))
    print("Area stdev: ", statistics.stdev(area_values))

def calculate_area(x, y):
    # From Mahdi
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
