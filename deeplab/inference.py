# Inference script

import os
import glob
from PIL import Image
from random import choice
import matplotlib.pyplot as plt
import torch
from torch.utils import data
from torchvision import models, transforms
from transforms_sample import *

val_dir = '/home/steveyang/projects/camelyon17/tile_images/deeplab/cross_val_0/val/'
images_dir = os.path.join(val_dir, 'images')
masks_dir = os.path.join(val_dir, 'masks')
mean =  (182.5253448486328, 182.49656677246094, 182.4678192138672)                                                                                                  
std =  (36.937557220458984, 36.780677795410156, 36.90703582763672)                                                                                                  

checkpoint_path = '/home/steveyang/projects/camelyon17/deeplab/checkpoints/deeplab_resnet50_experiment_5_100.pth'
model = models.segmentation.deeplabv3_resnet50(num_classes=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model']) 

images_files = os.listdir(images_dir)

def inference(images_files):
    filename = choice(images_files)
    input_image = Image.open(os.path.join(images_dir, filename))
    input_mask = Image.open(os.path.join(masks_dir, filename))

    preprocess = transforms.Compose([
                            Resize(256),
                            ToTensor(),
                            Normalize(mean=mean, std=std),])
    
    file_segmentation = FileSegmentation(val_dir, filename, transform=preprocess)
    file_loader = torch.utils.data.DataLoader(file_segmentation, batch_size=1)
    file_iter = iter(file_loader)
    input, labels = next(file_iter)
    model.eval()

    with torch.no_grad():
        output = model(input)['out']
        output_predictions = output.argmax(1)
        output_predictions = output_predictions.squeeze(0)
        labels = labels.squeeze(0)

    pred = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
    gt = Image.fromarray(labels.byte().cpu().numpy()).resize(input_image.size)

    plt.subplot(2, 2, 3)

    plt.subplot(221)
    plt.title("Prediction")
    plt.imshow(pred)

    plt.subplot(222)
    plt.title("Ground truth")
    plt.imshow(gt)

    plt.subplot(223)
    plt.title("Original image")
    plt.imshow(input_image)

    plt.show()


class FileSegmentation(data.Dataset):                                                                                                                                  
    def __init__(self, folder_path, filename, transform=None):
        super(FileSegmentation, self).__init__()                                                                                                                       
        self.transform = transform                                                                                                                                        
        self.img_path = os.path.join(folder_path, 'images', filename)
        self.mask_path = os.path.join(folder_path, 'masks', filename) 
                                                                                                                                                                                                                                              
    def __getitem__(self, index):                                                                                                                                         
        temp_image = Image.open(self.img_path).copy()
        temp_label = Image.open(self.mask_path).copy()                                                                                                                         
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
        return len(self.img_path)
