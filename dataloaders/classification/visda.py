
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import numpy as np
import torch
import torchvision.transforms as tvT


from PIL import Image


def make_dataset(image_list, labels):
    if labels:
      len_ = len(image_list)
      images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
      if len(image_list[0].split()) > 2:
        images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
      else:
        images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images

def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')


class VisDA_C(data.Dataset):
    def __init__(self, root_dir: str, split: str, mode:str = "RGB", labels=None, aug_mult=4,
                    hard_augment="randaugment") -> None:
        
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.aug_mult = aug_mult
        self.eval_transform = transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop((224, 224)),
                                            transforms.ToTensor(), transforms.Normalize(mean=self.mean, std=self.std)])
        self.train_transform = transforms.Compose([transforms.Resize((256, 256)),
                                            tvT.RandomCrop((224, 224)),
                                            tvT.RandomHorizontalFlip(),
                                            transforms.ToTensor(), transforms.Normalize(mean=self.mean, std=self.std)])

        if hard_augment == "randaugment":
            self.hard_augment = tvT.Compose([
                tvT.Lambda(lambda x : (x*255.).to(torch.uint8)),
                tvT.RandomHorizontalFlip(),
                tvT.RandAugment(),
                tvT.Lambda(lambda x : (x.to(torch.float32)/255.)),
            ])
        else:
            self.hard_augment = tvT.Compose([
                tvT.Lambda(lambda x : (x*255.).to(torch.uint8)),
                tvT.RandomHorizontalFlip(),
                tvT.AutoAugment(),
                tvT.Lambda(lambda x : (x.to(torch.float32)/255.)),
            ])

        if not os.path.exists(os.path.join(root_dir, split)):
            self.download(root_dir, split)
        else:
            print("Files Ready !")

        folder = os.path.join(root_dir, split)
        image_list = folder + "/image_list.txt"
        image_list = open(image_list).readlines()
        image_list = [folder + '/' + s for s in image_list]

        self.imgs = make_dataset(image_list, labels)
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

        self.random_indices = np.arange(len(self.imgs))
        np.random.shuffle(self.random_indices)


    def download(self, root_dir, split):
        os.makedirs(root_dir, exist_ok=True)

        if split=="train":
            os.system(f"wget http://csr.bu.edu/ftp/visda17/clf/train.tar --directory-prefix={root_dir}")
            os.system(f"tar -xvf {root_dir}/train.tar -C {root_dir}")
        else:
            os.system(f"wget http://csr.bu.edu/ftp/visda17/clf/validation.tar --directory-prefix={root_dir}")
            os.system(f"tar -xvf {root_dir}/validation.tar -C {root_dir}")
        

    def __len__(self):
        return len(self.imgs)


    def __getitem__(self, index):
        path, target = self.imgs[self.random_indices[index]]
        img = self.loader(path)
        imgs = torch.stack([self.eval_transform(img)] + [self.train_transform(img) for i in range(self.aug_mult)], dim=0)

        return imgs, target, index
