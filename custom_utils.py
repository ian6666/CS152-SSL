import json
import numpy as np
import cv2
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data import Dataset

path = 'ClimbingHoldDetection-15/train'

with open('ClimbingHoldDetection-15/train/_annotations.coco.json') as f:
    file = json.loads(f.read())
    images = file['images']
    annotations = file['annotations']

def extract_bbox(image, bbox_coord):
    x, y, w, h = [int(b) for b in bbox_coord]
    return image[y:y+h, x:x+w]


def parse_annotations(annotations):
    img_id_to_annotations = {}

    for a in annotations:
        if a['image_id'] in img_id_to_annotations:
            img_id_to_annotations[a['image_id']].append(a['id'])
        else:
            img_id_to_annotations[a['image_id']] = [a['id']]

    return img_id_to_annotations

def getBoudingBoxForImage(imageId, img_id_to_annotations, annotations, images, path = 'ClimbingHoldDetection-15/train', saving_dir=None):
    annotation_ids = img_id_to_annotations[imageId]
    img_path =  os.path.join(path, images[imageId]['file_name'])

    image = cv2.imread(img_path)


    for a_id in annotation_ids:
        extracted_img = extract_bbox(image, annotations[a_id]['bbox'])
        if saving_dir:
            cv2.imwrite(os.path.join(saving_dir, f"{imageId:05d}_{a_id:05d}.png"), extracted_img)

def extractAllImages(interval, img_id_to_annotations, annotations, images, path = 'ClimbingHoldDetection-15/train', saving_dir=None):
    from tqdm import tqdm
    for i in tqdm(interval):
        getBoudingBoxForImage(i, img_id_to_annotations, annotations, images, saving_dir="extractedLabeledDataset")

class CustomImageDataset(Dataset):
    def __init__(self, annotations, img_dir, transform=None):
        # self.img_labels = pd.read_csv(annotations_file)
        self.annotations = annotations
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        imageId = self.annotations[idx]['image_id']
        img_path = os.path.join(self.img_dir, f"{imageId:05d}_{idx:05d}.png")
        
        # error handling
        if not os.path.exists(img_path):
            print(f"File not found: {img_path}")
            return None, None
        
        image = read_image(img_path)
        label = self.annotations[idx]['category_id']

        if self.transform:
            image = self.transform(image)

        return image, label

def get_data(img_dir, annotations, batch_size=2, shuffle=False):
    transform = transforms.Compose([
        transforms.Resize((64, 64))
    ])
    trainDataset = CustomImageDataset(annotations, img_dir, transform=transform)
    train_dataloader = DataLoader(trainDataset, batch_size=batch_size, shuffle=shuffle)
    train_images, train_labels = next(iter(train_dataloader))
    return train_dataloader

def get_annotations():
    return annotations