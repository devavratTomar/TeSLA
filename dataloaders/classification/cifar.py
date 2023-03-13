import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import torchvision.transforms as tvT


class CIFAR_C(data.Dataset):
    def __init__(self, root_dir: str, dataset_name: str, corruption: str, level: int, aug_mult : int = 4) -> None:
        if dataset_name == 'cifar10':
            self.dataset = torchvision.datasets.CIFAR10(os.path.join(root_dir, ".."), train=False, download=True)
            self.mean = (0.4914, 0.4822, 0.4465)
            self.std = (0.2023, 0.1994, 0.2010)
        else:
            self.dataset = torchvision.datasets.CIFAR100(os.path.join(root_dir, ".."), train=False, download=True)
            self.mean = (0.5071, 0.4867, 0.4408)
            self.std = (0.2675, 0.2565, 0.2761)

        self.aug_mult = aug_mult
        self.eval_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=self.mean, std=self.std)])

        self.train_transform = tvT.Compose([
            transforms.ToTensor(),
            tvT.Resize(36),
            tvT.RandomResizedCrop(32, (0.8, 1)),
            tvT.RandomHorizontalFlip(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

        self.hard_augment = tvT.Compose([
            tvT.Resize(36),
            tvT.RandomResizedCrop(32, (0.8, 1)),
            tvT.RandomHorizontalFlip(),
            tvT.ColorJitter(0.5, 0.5, 0.5, 0.5)
        ])
        

        # corruption and level
        if "10.1" not in corruption:
            data = np.load(os.path.join(root_dir, corruption + ".npy"))
            self.dataset.data = data[(level-1) * 10000 : (level * 10000)]
        else:
            data_v4 = np.load(os.path.join(root_dir, corruption + "_v4_data.npy"))
            labels_v4 = np.load(os.path.join(root_dir, corruption + "_v4_labels.npy"))
            data_v6 = np.load(os.path.join(root_dir, corruption + "_v6_data.npy"))
            labels_v6 = np.load(os.path.join(root_dir, corruption + "_v6_labels.npy"))

            data = np.concatenate((data_v4, data_v6), axis=0)
            labels = np.concatenate((labels_v4, labels_v6))
            
            self.dataset.data = data
            self.dataset.targets = labels.tolist()

    def __len__(self):
        return len(self.dataset.targets)

    def __getitem__(self, index):
        img = self.dataset.data[index]
        label = self.dataset.targets[index]

        imgs = torch.stack([self.eval_transform(img)] + [self.train_transform(img) for i in range(self.aug_mult)], dim=0)
        label = torch.tensor(label, dtype=torch.long)

        return imgs, label, index


class CIFAR(data.Dataset):
    def __init__(self, root_dir: str, dataset_name: str, train:bool) -> None:
        if dataset_name == "cifar10":
            self.dataset = torchvision.datasets.CIFAR10(root_dir, train=train, download=True)
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2023, 0.1994, 0.2010)
        else:
            self.dataset = torchvision.datasets.CIFAR100(root_dir, train=train, download=True)
            mean = (0.5071, 0.4867, 0.4408)
            std = (0.2675, 0.2565, 0.2761)
        
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

        self.easy_augment = tvT.Compose([
            tvT.Resize(36),
            tvT.RandomResizedCrop(32, (0.8, 1)),
            tvT.RandomHorizontalFlip()
        ])

        self.hard_augment = tvT.Compose([
            tvT.Resize(36),
            tvT.RandomResizedCrop(32, (0.8, 1)),
            tvT.RandomHorizontalFlip(),
            tvT.ColorJitter(0.5, 0.5, 0.5, 0.5)
        ])

    def __len__(self):
        return len(self.dataset.targets)


    def __getitem__(self, index):
        img = self.dataset.data[index]
        label = self.dataset.targets[index]

        img = self.transform(img)
        label = torch.tensor(label, dtype=torch.long)

        return img, label, index
        