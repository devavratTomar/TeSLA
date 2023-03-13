
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import numpy as np
import json
from PIL import Image

def make_dataset(root_dir, corruption, level):
    id_map = {}  # map class_id to class_name, e.g., "0" -> "tench"
    word_map = {} # map word_id to class_id, e.g., "n01440764“ -> "0"
    
    with open(os.path.join(root_dir, "imageNet_labels.json")) as labels:
        labels = json.load(labels)
        for class_id, (word_id, class_name) in labels.items():
            id_map[class_id] = class_name
            word_map[word_id] = class_id

    images = []
    try:
        parent_folder = os.path.join(root_dir, corruption, str(level))
        if os.path.exists(parent_folder):
            for root, _, fnames in sorted(os.walk(parent_folder, followlinks=True)):
                for fname in fnames:
                    path = os.path.join(root, fname)
                    label = int(word_map[root.split("/")[-1]])
                    images.append((path, label))
    except:
        raise ValueError(f"Cannot find folder {parent_folder}")

    return images, id_map

def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')

class IMAGENET_C(data.Dataset):
    def __init__(self, root_dir:str, corruption:str = "brightness", level:int = 5, loader:str = "RGB", aug_mult=4):
        
        # (1) mean/std
        self.mean = (0.485, 0.456, 0.406)   # same as imageNet
        self.std = (0.229, 0.224, 0.225)
        self.aug_mult = aug_mult

        # (2) transforms
        self.eval_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=self.mean, std=self.std)])
        self.train_transform = transforms.Compose([
                                            transforms.Resize((256, 256)),
                                            transforms.RandomResizedCrop((224, 224), (0.8, 1)),
                                            transforms.RandomHorizontalFlip(),
                                            # transforms.RandomApply(
                                            #     [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                                            #     p=0.8
                                            # ),
                                            transforms.ToTensor(), transforms.Normalize(mean=self.mean, std=self.std)])
        self.hard_augment = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop((224, 224), (0.8, 1)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.5, 0.5, 0.5, 0.5)
        ])

        # (3) make_dataset (path, label)
        self.imgs = []
        self.imgs, self.id_map = make_dataset(root_dir, corruption, level)
        
        # (4) other attributes
        if loader == 'RGB':
            self.loader = rgb_loader
        elif loader == 'L':
            self.loader = l_loader

        self.imgs_size = len(self.imgs)
        self.random_indices = np.arange(len(self.imgs))
        np.random.shuffle(self.random_indices)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        path, label = self.imgs[self.random_indices[index % self.imgs_size]]  # make sure index is within then range
        img = self.loader(path)
        imgs = torch.stack([self.eval_transform(img)] + [self.train_transform(img) for i in range(self.aug_mult)], dim=0)

        return imgs, label, index
