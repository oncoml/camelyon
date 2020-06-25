# Training Camelyon17 images on Deeplab with Pytorch

import numpy as np
import glob
import os
import torch
from torch import nn, optim
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torch.utils import data
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import utils
from utils import MetricLogger, ConfusionMatrix
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
from transforms_sample import Resize, RandomVerticalFlip, RandomHorizontalFlip, ToTensor, Normalize
from sklearn.metrics import matthews_corrcoef
from crfseg import CRF

train_dir = '/home/steveyang/projects/camelyon17/tile_images/deeplab/cross_val_neg/train'
val_dir = '/home/steveyang/projects/camelyon17/tile_images/deeplab/cross_val_neg/val'

train_mean =  (182.5253448486328, 182.49656677246094, 182.4678192138672)
train_std =  (36.937557220458984, 36.780677795410156, 36.90703582763672)
val_mean =  (182.54859924316406, 182.52549743652344, 182.4474334716797)
val_std =  (37.22926330566406, 37.0758056640625, 37.21209716796875)

def load_data(train_dir, val_dir):
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    train_transforms = transforms.Compose([Resize(256),
                                        RandomHorizontalFlip(0.5),
                                        RandomVerticalFlip(0.5), 
                                        ToTensor(),
                                        Normalize(train_mean, train_std)])
    val_transforms = transforms.Compose([Resize(256),
                                        ToTensor(),
                                        Normalize(val_mean, val_std)])

    train_data = DatasetSegmentation(train_dir, transform=train_transforms)
    val_data = DatasetSegmentation(val_dir, transform=val_transforms)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=16, shuffle=True)

    return train_loader, val_loader


def train(train_dir, val_dir, checkpoint_file=None):
    train_loader, val_loader = load_data(train_dir, val_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=2)

    for param in model.parameters():
        param.requires_grad = True

    model = nn.Sequential(model, CRF(n_spatial_dims=2)) 

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.002)
    model.to(device)

    writer = SummaryWriter('runs/deeplab_experiment_11')

    epochs = 100 
    print_freq = 500
    save_freq = 25
    
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda x: (1 - x / (len(train_loader) * epochs)) ** 0.9)    

    if checkpoint_file is not None:
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        last_epoch = checkpoint['epoch']
        acc_global = checkpoint['acc_global']
        acc = checkpoint['acc']
        iou = checkpoint['iou']
        miou = checkpoint['miou']
        dice_coef = checkpoint['dice_coef']
        mcc = checkpoint['mcc']
    else:
        last_epoch = 0 

    for epoch in range(last_epoch+1, last_epoch+epochs+1):
        train_mcc = 0.0
        train_confmat = utils.ConfusionMatrix(num_classes=2)

        for inputs, labels in train_loader: 
            model.train()
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            try:
                logps = model(inputs)
                #logps = logps['out'].squeeze(1)
            except:
                continue

            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step(loss.cpu().data.numpy())
            
            train_pred = torch.argmax(logps, dim=1)
            train_confmat.update(labels.flatten().long(), train_pred.flatten())
            train_mcc += matthews_corrcoef(labels.cpu().numpy().flatten(), train_pred.cpu().numpy().flatten())
           
        train_acc_global, train_acc, train_iou, train_miou = train_confmat.compute()
        train_mcc = train_mcc/len(train_loader)
        print("Train loss: ", loss.item(), " ... ", "Train acc: ", train_acc_global, " ... ", 
                "Train mIOU: ", train_miou, " ... ", "Train MCC: ", train_mcc)

        writer.add_scalar('Train/Loss', loss.item(), epoch)
        writer.add_scalar('Train/Global Accuracy', train_acc_global, epoch)
        writer.add_scalar('Train/Accuracy/nontumor', train_acc[0], epoch)
        writer.add_scalar('Train/Accuracy/tumor', train_acc[1], epoch)
        writer.add_scalar('Train/IoU/nontumor', train_iou[0], epoch)
        writer.add_scalar('Train/IoU/tumor', train_iou[1], epoch)
        writer.add_scalar('Train/mIoU', train_miou, epoch)
        writer.add_scalar('Train/MCC', train_mcc, epoch)

        # Evaluate validation loss and confusion matrix after every epoch
        model.eval()
        val_loss = 0.0
        val_mcc = 0.0
        confmat = utils.ConfusionMatrix(num_classes=2)

        with torch.no_grad():                                                                                                                                     
            for val_inputs, val_labels in val_loader:                                                                                                             
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)                                                                             

                try:
                    val_logps = model(val_inputs)
                    #val_logps = val_logps['out'].squeeze(1)
                except:
                    continue

                val_preds = torch.argmax(val_logps, dim=1)
                probability = torch.sigmoid(val_logps)
                predicted = (probability > 0.5).int()
                confmat.update(val_labels.flatten().long(), val_preds.flatten())
                val_mcc += matthews_corrcoef(val_labels.cpu().numpy().flatten(), val_preds.cpu().numpy().flatten())
                batch_loss = criterion(val_logps, val_labels)
                val_loss += batch_loss.item()     
        
        val_loss = val_loss/len(val_loader)
        acc_global, acc, iou, miou = confmat.compute()
        dice_coef = dice(val_preds.flatten(), val_labels.flatten())
        val_mcc = val_mcc/len(val_loader)
        print("Val loss: ", val_loss, " ... ", "Val acc: ", acc_global, " ... ", 
                "Val mIOU: ", miou, " ... ", "Val Dice coeff: ", dice_coef,
                "Val MCC: ", val_mcc)
    
        writer.add_scalar('Val/Loss', val_loss, epoch)
        writer.add_scalar('Val/Global Accuracy', acc_global, epoch)
        writer.add_scalar('Val/Accuracy/nontumor', acc[0], epoch)
        writer.add_scalar('Val/Accuracy/tumor', acc[1], epoch)
        writer.add_scalar('Val/IoU/nontumor', iou[0], epoch)
        writer.add_scalar('Val/IoU/tumor', iou[1], epoch)
        writer.add_scalar('Val/mIoU', miou, epoch)
        writer.add_scalar('Val/Dice coeff', dice_coef, epoch)
        writer.add_scalar('Val/MCC', val_mcc, epoch)
        writer.close()

        # Save checkpoint after every save_freq epochs
        if epoch % save_freq == 0:
            utils.save_on_master(
                {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'acc_global': acc_global,
                    'acc': acc,
                    'iou': iou,
                    'miou': miou,
                    'dice_coef': dice_coef,
                    'mcc': val_mcc
                },
                os.path.join('./checkpoints', 'deeplab_resnet101_experiment_11_{}.pth'.format(epoch)))


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


class DatasetSegmentation(data.Dataset):
    def __init__(self, folder_path, transform=None):
        super(DatasetSegmentation, self).__init__()
        self.transform = transform
        self.img_files = glob.glob(os.path.join(folder_path, 'images', '*.png'))
        self.mask_files = []
        for img_path in self.img_files:
            self.mask_files.append(os.path.join(folder_path, 'masks', os.path.basename(img_path)))

    def __getitem__(self, index):
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]
        temp_image = Image.open(img_path).copy()
        temp_label = Image.open(mask_path).copy()

        if self.transform:
            sample = {'image': temp_image, 'labels': temp_label}
            t_sample = self.transform(sample)
            return t_sample['image'].float(), t_sample['labels'].long()

        else:
            temp_image = np.array(temp_image)
            temp_image = temp_data.transpose(2, 0, 1)
            temp_label = np.array(temp_label)
            pattern = np.array([0, 0, 0])
            mask = (temp_label == pattern).all(axis=2)
            temp_label = np.where(mask, 0, 1) # Black [0, 0, 0] labeled as 0; white as 1
            return torch.from_numpy(temp_image).float(), torch.from_numpy(temp_label).long()

    def __len__(self):
        return len(self.img_files)


def dice(predicted, labels):
    smooth = 1.
    intersection = (predicted * labels).sum()

    return (2. * intersection + smooth) / (predicted.sum() + labels.sum() + smooth)
