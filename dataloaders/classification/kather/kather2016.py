import torchvision
import numpy as np
import shutil
import random, os, os.path as osp
import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import argparse

import torch
import torchvision.transforms as transforms

def create_val_img_folder(path_data_in='data/Kather_texture_2016_image_tiles_5000'):
    '''
    This method is responsible for separating Kather 2016 images into separate sub folders
    path_data_in points to the uncompressed folder resulting from downloading from zenodo:
    https://zenodo.org/record/53169#.YgIcsMZKjCI the file Kather_texture_2016_image_tiles_5000.zip
    '''
    path_data = path_data_in.replace('Kather_texture_2016_image_tiles_5000', 'kather2016')
    path_train_data = osp.join(path_data, 'train')
    path_val_data = osp.join(path_data, 'val')
    path_test_data = osp.join(path_data, 'test')

    os.makedirs(osp.join(path_train_data), exist_ok=True)
    os.makedirs(osp.join(path_val_data), exist_ok=True)
    os.makedirs(osp.join(path_test_data), exist_ok=True)

    subfs = os.listdir(path_data_in)
    # loop over subfolders in train data folder
    for f in subfs:
        os.makedirs(osp.join(path_train_data, f), exist_ok=True)
        os.makedirs(osp.join(path_val_data, f), exist_ok=True)
        os.makedirs(osp.join(path_test_data, f), exist_ok=True)

        path_this_f = osp.join(path_data_in, f)
        im_list = os.listdir(path_this_f)

        val_len = int(0.15 * len(im_list))  # random 15% for val
        test_len = int(0.15 * len(im_list))  # random 15% for test

        # take 15% out of im_list for test
        im_list_test = random.sample(im_list, test_len)
        im_list_train = list(set(im_list) - set(im_list_test))

        # take 15% out of im_list for val, the rest is train
        im_list_val = random.sample(im_list_train, val_len)
        im_list_train = list(set(im_list_train) - set(im_list_val))

        # loop over subfolders and move train/val/test images over to corresponding subfolders
        for im in im_list_train:
            shutil.move(osp.join(path_data_in, f, im), osp.join(path_train_data, f, im))
        for im in im_list_val:
            shutil.move(osp.join(path_data_in, f, im), osp.join(path_val_data, f, im))
        for im in im_list_test:
            shutil.move(osp.join(path_data_in, f, im), osp.join(path_test_data, f, im))

class Kather2016(torchvision.datasets.ImageFolder):
    def __init__(self, root, aug_mult):
        super(Kather2016, self).__init__(root)

        self.mean, self.std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        self.aug_mult=aug_mult

        self.train_transform = transforms.Compose([
            transforms.Resize((150, 150)),
            transforms.RandomCrop(150, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

        self.eval_transform = transforms.Compose([
            transforms.Resize((150, 150)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    def __getitem__(self, index):
        img, label = super().__getitem__(index)

        imgs = torch.stack([self.eval_transform(img)] + [self.train_transform(img) for i in range(self.aug_mult)], dim=0)
        label = torch.tensor(label, dtype=torch.long)

        return imgs, label, index

def subsample_dataset(dataset, idxs):
    # dataset.imgs = [x for i, x in enumerate(dataset.imgs) if i in idxs]
    # dataset.samples = [x for i, x in enumerate(dataset.samples) if i in idxs]
    # I am replacing the above awful code by a faster version, no need to enumerate for each idx the same set again and
    # again, people should be more careful with comprehensions (agaldran).
    imgs, sampls = [], []
    for i in idxs:
        imgs.append(dataset.imgs[i])
        sampls.append(dataset.samples[i])
    dataset.imgs = imgs
    dataset.samples = sampls
    dataset.targets = np.array(dataset.targets)[idxs].tolist()

    return dataset

def subsample_classes(dataset, include_classes=(0,1)):
    _ = enumerate(dataset.targets)
    cls_idxs = [x for x, t in _ if t in include_classes]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)
    dataset.target_transform = lambda x: target_xform_dict[x]

    # torchvision ImageFolder dataset have a handy class_to_idx attribute that we have spoiled and need to re-do
    # filter class_to_idx to keep only include_classes
    new_class_to_idx = {key: val for key, val in dataset.class_to_idx.items() if val in target_xform_dict.keys()}
    # fix targets so that they start in 0 and are correlative
    new_class_to_idx = {k: target_xform_dict[v] for k, v in new_class_to_idx.items()}
    # replace bad class_to_idx with good one
    dataset.class_to_idx = new_class_to_idx

    # and let us also add a idx_to_class attribute
    dataset.idx_to_class = dict((v, k) for k, v in new_class_to_idx.items())

    return dataset

def get_equal_len_datasets(dataset1, dataset2):
    """
    Make two datasets the same length
    """

    if len(dataset1) > len(dataset2):

        rand_idxs = np.random.choice(range(len(dataset1)), size=(len(dataset2, )))
        subsample_dataset(dataset1, rand_idxs)

    elif len(dataset2) > len(dataset1):

        rand_idxs = np.random.choice(range(len(dataset2)), size=(len(dataset1, )))
        subsample_dataset(dataset2, rand_idxs)

    return dataset1, dataset2

def get_kather2016_datasets(root_dir, known_classes=(0,1), seed=0, aug_mult=0):


    kather_test_dir = osp.join(root_dir, 'test')

    np.random.seed(seed)

    # Build test dataset and subsample known/unknown classes
    test_dataset = Kather2016(root=kather_test_dir, aug_mult=aug_mult)
    test_dataset = subsample_classes(test_dataset, include_classes=known_classes)

    return test_dataset


if __name__ == '__main__':

    parser = argparse.ArgumentParser("Training")
    parser.add_argument('--path_data_in', type=str, default='data/Kather_texture_2016_image_tiles_5000', help="")

    args = parser.parse_args()
    create_val_img_folder(args.path_data_in)
