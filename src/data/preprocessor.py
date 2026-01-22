import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import csv
import ast
import re
from torch.utils.data import DataLoader



CRASH_IMG_FOLDER=r"e:\Kush\2nd_year\projects\accident_pred\dataset\frames\Crash-1500"
NORMAL_IMG_FOLDER=r"e:\Kush\2nd_year\projects\accident_pred\dataset\frames\Normal-3000"

def image_resize_toTensor(IMG_FOLDER,filename):
    image=cv2.imread(os.path.join(IMG_FOLDER, filename))
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(img_rgb, (224, 224))
    image_resized = image_resized.astype("float32") / 255.0
    image_tensor = np.transpose(image_resized, (2, 0, 1))
    image_tensor = torch.from_numpy(image_tensor)
    return image_tensor

def get_images_labelled(images, label=[]):
    images_name_labeled=[]
    if len(label)==0:
        for image in images:
            images_name_labeled.append((image,0))
        return images_name_labeled
    else:
        for image,label in zip(images,label):
            images_name_labeled.append((image,label))
        return images_name_labeled



class FrameDataset(Dataset):
    def __init__(self, frame_dir, annotations, transform=None):
        self.frame_dir = frame_dir
        self.annotations = annotations
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        filename, label = self.annotations[idx]
        image=image_resize_toTensor(self.frame_dir,filename)
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long)
        return image, label






def read_crash_labels(crash_labels_path):
    """Read crash labels from file and return flattened 1D array."""
    crash_frame_label_list = []
    with open(crash_labels_path, 'r') as file:
        for line in file:
            match = re.search(r'\[(.*?)\]', line)
            if match:
                list_str = '[' + match.group(1) + ']'
                feature_list = ast.literal_eval(list_str)
                crash_frame_label_list.append(feature_list)
    
    crash_frame_label_array = np.array(crash_frame_label_list)
    return crash_frame_label_array.flatten()


def get_image_filenames(folder_path):
    """Get sorted list of jpg filenames from a folder."""
    return sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')])