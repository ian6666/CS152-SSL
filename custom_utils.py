import json
import numpy as np
import cv2
import os
import pandas as pd
import torch
from torchvision.io import read_image
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.nn.functional import one_hot

labeled_path = 'ClimbingHoldDetection-15/train'
with open('ClimbingHoldDetection-15/train/_annotations.coco.json') as f:
    labeled_file = json.loads(f.read())
    labeled_images = labeled_file['images']
    labeled_annotations = labeled_file['annotations']

unlabeled_path = "Climbing-Holds-and-Volumes-14/train/"
with open("Climbing-Holds-and-Volumes-14/train/_annotations.coco.json") as f:
    unlabeled_file = json.loads(f.read())
    unlabeled_images = unlabeled_file['images']
    unlabeled_annotations = unlabeled_file['annotations']
    

# Used to make new json file
def deleteCategoryID(annotations, category_id):
    new_annotations = []
    id = 0
    for i in range(len(annotations)):
        if int(annotations[i]['category_id']) != category_id:
            annotations[i]['id'] = id
            new_annotations.append(annotations[i])
            id += 1
          
    return new_annotations


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

def getBoudingBoxForImage(imageId, img_id_to_annotations, annotations, images, path, saving_dir=None):
    # import pdb
    # pdb.set_trace()
    annotation_ids = img_id_to_annotations[imageId]
    img_path =  os.path.join(path, images[imageId]['file_name'])

    image = cv2.imread(img_path)


    for a_id in annotation_ids:
        extracted_img = extract_bbox(image, annotations[a_id]['bbox'])
        if saving_dir:
            cv2.imwrite(os.path.join(saving_dir, f"{imageId:05d}_{a_id:05d}.png"), extracted_img)

def extractAllImages(interval, img_id_to_annotations, annotations, images, path, saving_dir=None):
    from tqdm import tqdm
    for i in tqdm(interval):
        if i in img_id_to_annotations:
            getBoudingBoxForImage(i, img_id_to_annotations, annotations, images, path, saving_dir=saving_dir)
        # else:
        #     print(str(i)+ " not in annotations")



class LabeledImageDataset(Dataset):
    def __init__(self, annotations, img_dir, num_classes, transform=None):
        # self.img_labels = pd.read_csv(annotations_file)
        self.annotations = annotations
        self.img_dir = img_dir
        self.transform = transform
        self.num_classes = num_classes
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        imageId = self.annotations[idx]['image_id']
        img_path = os.path.join(self.img_dir, f"{imageId:05d}_{idx:05d}.png")
        
        image = read_image(img_path)
        label = self.annotations[idx]['category_id']

        if self.transform:
            image = self.transform(image)

        # one_hot_label = one_hot(torch.tensor(label))
        return image, label-1

class UnlabeledImageDataset(Dataset):
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
        
        image = read_image(img_path)

        if self.transform:
            image = self.transform(image)

        return image
    
def get_labeled_data(img_dir, annotations, batch_size=32, shuffle=False):
    transform = transforms.Compose([
        transforms.Resize((64, 64))
    ])
    labeled_dataset = LabeledImageDataset(annotations, img_dir, 5, transform=transform)
    labeled_dataloader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=shuffle)
    return labeled_dataloader

def get_unlabeled_data(img_dir, annotations, batch_size=32, shuffle=False):
    transform = transforms.Compose([
        transforms.Resize((64, 64))
    ])
    unlabeled_dataset = UnlabeledImageDataset(annotations, img_dir, 5, transform=transform)
    unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=shuffle)
    return unlabeled_dataloader

def get_annotations():
    with open('ClimbingHoldDetection-15/train/_annotations.coco.json') as f:
        file = json.loads(f.read())
        annotations = file['annotations']
        return annotations

def augmentAndPredict(model, images , k, num_classes, transforms):
    """
    model: model
    images: a batch of images
    k: k transforms
    transforms: a list of transforms legnth k
    """
    batch_size, dims, height, width = images.shape
    all_augmented_inputs = torch.zeros([k*batch_size, dims, height, width])
    all_preds =  torch.zeros([k*batch_size, num_classes])
    for t in range(k):
        transformed_images = transforms[t](images)
        preds = model(transformed_images)
        all_augmented_inputs[t*batch_size:(t+1)*batch_size] = transformed_images
        all_preds[t*batch_size:(t+1)*batch_size] = preds
    
    return all_augmented_inputs, all_preds

if __name__ == "__main__":
    labeled_img_id_to_annotations=parse_annotations(labeled_annotations)
    labeled_data_saving_dir = "LabeledData"
    num_labeled_images = len(labeled_images)
    extractAllImages(range(num_labeled_images), labeled_img_id_to_annotations, labeled_annotations, labeled_images, path = labeled_path, saving_dir=labeled_data_saving_dir)


    unlabeled_img_id_to_annotations=parse_annotations(unlabeled_annotations)
    unlabeled_data_saving_dir = "UnlabeledData"
    num_unlabeled_images = len(unlabeled_images)
    extractAllImages(range(num_unlabeled_images), unlabeled_img_id_to_annotations, unlabeled_annotations, unlabeled_images, path = unlabeled_path, saving_dir=unlabeled_data_saving_dir)

