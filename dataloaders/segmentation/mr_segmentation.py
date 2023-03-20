import numpy as np
import os
import re

import torch
import torch.utils.data as data
import albumentations as A

from utilities.utils import natural_sort

DATASET_MODES = ['spinalcord', 'heart', 'prostate']


class TestTimeDataset(data.Dataset):
    def __init__(self, rootdir, sites, datasetmode):
        if datasetmode not in DATASET_MODES:
            raise Exception('Dataset not recognized')
        self.mean = [0.5]
        self.std = [0.5]
        
        if datasetmode == DATASET_MODES[0]:
            img_query = 'image'
            seg_query = 'mask'

        elif datasetmode == DATASET_MODES[1]:
            img_query = 'sa.nii.gz'
            seg_query = 'sa_gt.nii.gz'
        
        elif datasetmode == DATASET_MODES[2]:
            seg_query = 'segmentation'

        self.rootdir = rootdir
        self.sites = sites

        # sample dataset from the given sites
        if datasetmode == DATASET_MODES[2]:
            self.all_imgs = np.array(natural_sort([f for f in os.listdir(rootdir) if seg_query not in f and self.is_site(f)]))
        else:
            self.all_imgs = np.array(natural_sort([f for f in os.listdir(rootdir) if img_query in f and self.is_site(f)]))
        
        self.all_segs = np.array(natural_sort([f for f in os.listdir(rootdir) if seg_query in f and self.is_site(f)]))
        
        assert len(self.all_imgs) == len(self.all_segs)
        # self.filter_no_seg()

        self.augmentor = A.Compose([A.Resize(256, 256)])
        

    def filter_no_seg(self,):
        filtered_img = []
        filtered_seg = []
        
        n_filtered = 0
        for i in range(len(self.all_imgs)):
            if np.load(os.path.join(self.rootdir, self.all_segs[i])).sum() != 0:
                filtered_img.append(self.all_imgs[i])
                filtered_seg.append(self.all_segs[i])
                n_filtered += 1

        print("Filtered %d imgs out of %d" % (n_filtered, len(self.all_imgs)))

        self.all_imgs = filtered_img
        self.all_segs = filtered_seg


    def is_site(self, name):
        for site in self.sites:
            if site in name:
                return True
        
        return False
    
    def __getitem__(self, index):
        
        img = np.load(os.path.join(self.rootdir, self.all_imgs[index])).astype(np.float32)
        seg = np.load(os.path.join(self.rootdir, self.all_segs[index]))

        transformed = self.augmentor(image=img, mask=seg)

        img = transformed['image']
        img = 2*torch.from_numpy(img).to(torch.float32).unsqueeze(0) - 1

        seg = transformed['mask']
        seg = torch.from_numpy(seg).to(torch.long)
        
        return img, seg, img.shape, self.all_segs[index]

    
    def __len__(self):
        return len(self.all_imgs)


class TestTimeVolumeDataset(data.Dataset):
    def __init__(self, rootdir, sites, datasetmode):
        if datasetmode not in DATASET_MODES:
            raise Exception('Dataset not recognized')
        
        self.mean = [0.5]
        self.std = [0.5]
        
        if datasetmode == DATASET_MODES[0]:
            img_query = 'image'
            seg_query = 'mask'

        elif datasetmode == DATASET_MODES[1]:
            img_query = 'sa.nii.gz'
            seg_query = 'sa_gt.nii.gz'

        elif datasetmode == DATASET_MODES[2]:
            seg_query = 'segmentation'

        self.datasetmode = datasetmode
        self.rootdir = rootdir
        self.sites = sites
        
        # sample dataset from the given sites
        if datasetmode == DATASET_MODES[2]:
            all_imgs = natural_sort([f for f in os.listdir(rootdir) if seg_query not in f and self.is_site(f)])
        else:
            all_imgs = natural_sort([f for f in os.listdir(rootdir) if img_query in f and self.is_site(f)])
        
        all_segs = natural_sort([f for f in os.listdir(rootdir) if seg_query in f and self.is_site(f)])
        
        assert len(all_imgs) == len(all_segs)

        grouped_imgs = {}
        grouped_segs = {}

        # group the slices based on patient
        for img, seg in zip(all_imgs, all_segs):
            if self.datasetmode == DATASET_MODES[0]:
                assert img.split('-')[:2] == seg.split('-')[:2]
                patient_name = '-'.join(img.split('-')[:2])
            elif self.datasetmode == DATASET_MODES[2]:
                assert re.split('-|_|\.', img)[:3] == re.split('-|_|\.', seg)[:3]
                patient_name = seg.split("segmentation.")[0]
            else:
                assert img.split('_')[0] == seg.split('_')[0]
                patient_name = img.split('_')[0]
            
            if patient_name in grouped_imgs:
                grouped_imgs[patient_name].append(img)
                grouped_segs[patient_name].append(seg)

            else:
                grouped_imgs[patient_name] = [img]
                grouped_segs[patient_name] = [seg]

        self.augmentor = A.Compose([A.Resize(256, 256)])
        self.all_imgs = [v for _ , v in grouped_imgs.items()]
        self.all_segs = [v for _ , v in grouped_segs.items()]
    
    
    def is_site(self, name):
        for site in self.sites:
            if site in name:
                return True
        
        return False
    

    def __getitem__(self, index):
        img_names = self.all_imgs[index]
        seg_names = self.all_segs[index]
        out_imgs = []
        out_segs = []

        for i_n, s_n in zip(img_names, seg_names):    
            img = np.load(os.path.join(self.rootdir, i_n)).astype(np.float32)
            seg = np.load(os.path.join(self.rootdir, s_n))

            transformed = self.augmentor(image=img, mask=seg)

            img = transformed['image']
            img = 2*torch.from_numpy(img).to(torch.float32).unsqueeze(0) - 1

            seg = transformed['mask']
            seg = torch.from_numpy(seg).to(torch.long)
            
            out_imgs.append(img)
            out_segs.append(seg)
        
        out_imgs = torch.stack(out_imgs)
        out_segs = torch.stack(out_segs)

        return out_imgs, out_segs, out_imgs.shape, seg_names

    
    def __len__(self):
        return len(self.all_imgs)


class TrainTimeDataset(data.Dataset):
    def __init__(self, rootdir, sites, datasetmode):
        if datasetmode not in DATASET_MODES:
            raise Exception('Dataset not recognized')

        if datasetmode == DATASET_MODES[0]:
            img_query = 'image'
            seg_query = 'mask'

        elif datasetmode == DATASET_MODES[1]:
            img_query = 'sa.nii.gz'
            seg_query = 'sa_gt.nii.gz'
        
        elif datasetmode == DATASET_MODES[2]:
            seg_query = 'segmentation'

        self.rootdir = rootdir
        self.sites = sites

        # sample dataset from the given sites
        if datasetmode == DATASET_MODES[2]:
            self.all_imgs = np.array(natural_sort([f for f in os.listdir(rootdir) if seg_query not in f and self.is_site(f)]))
        else:
            self.all_imgs = np.array(natural_sort([f for f in os.listdir(rootdir) if img_query in f and self.is_site(f)]))
        
        self.all_segs = np.array(natural_sort([f for f in os.listdir(rootdir) if seg_query in f and self.is_site(f)]))
        
        assert len(self.all_imgs) == len(self.all_segs)
        
        self.augmenter =  A.Compose([
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.Rotate(p=0.5),
                A.GaussianBlur(),
                A.RandomBrightness(0.2),
                A.RandomContrast(0.2),
                A.RandomGamma(),
                A.RandomResizedCrop(256, 256, scale=(0.5, 1.0)),
            ])

    def is_site(self, name):
        for site in self.sites:
            if site in name:
                return True
        
        return False
    
    def __getitem__(self, index):
        
        img = np.load(os.path.join(self.rootdir, self.all_imgs[index])).astype(np.float32)
        seg = np.load(os.path.join(self.rootdir, self.all_segs[index]))

        transformed = self.augmenter(image=img, mask=seg)

        img = transformed['image']
        img = 2*torch.from_numpy(img).to(torch.float32).unsqueeze(0) - 1

        seg = transformed['mask']
        seg = torch.from_numpy(seg).to(torch.long)
        
        return img, seg, img.shape, self.all_segs[index]

    
    def __len__(self):
        return len(self.all_imgs)