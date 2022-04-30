import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import datasets
import pandas as pd
from PIL import Image
from skimage import io, img_as_ubyte

import moco.builder
import moco.loader
import moco.optimizer

np.random.seed(0)

class Dataset():
    def __init__(self, csv_file, transform=None):
        self.files_list = pd.read_csv(csv_file)
        self.transform = transform
    def __len__(self):
        return len(self.files_list)
    def __getitem__(self, idx):
        # get file name (row:idx, col:0)
        temp_path = self.files_list.iloc[idx, 0]
        # print(temp_path)
        # print("before img")
        img = Image.open(temp_path)
        # print("after img")
        img = transforms.functional.to_tensor(img)
        if self.transform:
            sample = self.transform(img)
        return sample


class ToPIL(object):
    def __call__(self, sample):
        img = sample
        img = transforms.functional.to_pil_image(img)
        return img 

class DataSetWrapper(object):

    def __init__(self, batch_size, num_workers, valid_size, input_shape, s, crop_min):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.s = s
        self.input_shape = eval(input_shape)
        self.crop_min = crop_min

    def get_data_loaders(self):
        # apply SimCLR transformation to dataset
        data_augment = self._get_simclr_pipeline_transform()
        train_dataset = Dataset(csv_file='TCGA_train.csv', transform=SimCLRDataTransform(data_augment))
        valid_dataset = Dataset(csv_file='TCGA_val.csv', transform=SimCLRDataTransform(data_augment))

        # apply MocoV3 transformation to dataset
        # data_augment_1 = self.get_mocov3_transform_1()
        # data_augment_2 = self.get_mocov3_transform_2()
        # train_dataset = Dataset(csv_file='new_train_temp.csv', transform=MocoV3DataTransform(data_augment_1, data_augment_2))
        # valid_dataset = Dataset(csv_file='new_val_temp.csv', transform=MocoV3DataTransform(data_augment_1, data_augment_2))

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  num_workers=self.num_workers, drop_last=True, shuffle=True, pin_memory=True)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size,
                                  num_workers=self.num_workers, drop_last=True, pin_memory=True)

        # train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader

    def _get_simclr_pipeline_transform(self):
        # get a set of data augmentation transformations as described in the SimCLR paper.
        # combination of random crop and color distortion is crucial to achieve a good performance

        # making random color rotations (s is the strength of color distortion)
        color_jitter = transforms.ColorJitter(0.8 * self.s, 0.8 * self.s, 0.8 * self.s, 0.2 * self.s)
        data_transforms = transforms.Compose([ToPIL(),
                                              transforms.RandomResizedCrop(size=(self.input_shape[0], self.input_shape[1])),
                                              # transforms.Resize((self.input_shape[0],self.input_shape[1])),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.06 * self.input_shape[0])),
                                              transforms.ToTensor()])
        return data_transforms
    
    # follow BYOL's augmentation recipe: https://arxiv.org/abs/2006.07733
    def get_mocov3_transform_1(self):
        data_transforms = transforms.Compose([ToPIL(),
                                              transforms.RandomResizedCrop(224, scale=(self.crop_min, 1.)),
                                              transforms.RandomApply([
                                                  transforms.ColorJitter(0.4, 0.4, 0.2, 0.1) # not strenghtened
                                              ], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              # transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=1.0),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225])
                                              ])

        return data_transforms
    
    def get_mocov3_transform_2(self):
        data_transforms = transforms.Compose([ToPIL(),
                                              transforms.RandomResizedCrop(224, scale=(self.crop_min, 1.)),
                                              transforms.RandomApply([
                                                  transforms.ColorJitter(0.4, 0.4, 0.2, 0.1) # not strenghtened
                                              ], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              # transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.1),
                                              transforms.RandomApply([moco.loader.Solarize()], p=0.2),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225])
                                              ])

        return data_transforms

    


class SimCLRDataTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        xi = self.transform(sample)
        xj = self.transform(sample)
        return xi, xj


class MocoV3DataTransform(object):

    """Take two random crops of one image"""

    def __init__(self, transform1, transform2):
        self.transform1 = transform1
        self.transform2 = transform2
    
    def __call__(self, sample):
        xi = self.transform1(sample)
        xj = self.transform2(sample)
        return xi, xj
