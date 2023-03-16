"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the OLTR project which
notice below and in LICENSE in the root directory of
this source tree.

Copyright (c) 2019, Zhongqi Miao
All rights reserved.
"""

import json
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import os
from PIL import Image
from data.ImbalanceCIFAR import IMBALANCECIFAR10, IMBALANCECIFAR100
from torch.utils.data import DistributedSampler
import torch_xla.core.xla_model as xm

import numpy as np
import torch
from tqdm import tqdm


# Image statistics
RGB_statistics = {
    'iNaturalist18': {
        'mean': [0.466, 0.471, 0.380],
        'std': [0.195, 0.194, 0.192]
    },
    'default': {
        'mean': [0.485, 0.456, 0.406],
        'std':[0.229, 0.224, 0.225]
    }
}

# Data transformation with augmentation
def get_data_transform(split, rgb_mean, rbg_std, key='default'):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ]) if key == 'iNaturalist18' else transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ])
    }
    return data_transforms[split]


# Dataset
class LT_Dataset(Dataset):

    def __init__(self, root, txt, dataset, transform=None, meta=False):
        self.img_path = []
        self.labels = []
        self.transform = transform

        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))

        # save the class frequency
        if 'train' in txt and not meta:
            if not os.path.exists('cls_freq'):
                os.makedirs('cls_freq')
            freq_path = os.path.join('cls_freq', dataset + '.json')
            self.img_num_per_cls = [0 for _ in range(max(self.labels)+1)]
            for cls in self.labels:
                self.img_num_per_cls[cls] += 1
            with open(freq_path, 'w') as fd:
                json.dump(self.img_num_per_cls, fd)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label, index

class MMAPDataset(Dataset):
    def __init__(self, root, txt, dataset,txt_split, transform=None, meta=False):
        super().__init__()
        self.img_path = []
        self.labels = []
        self.transform = transform
        self.xm = xm

        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))

        # save the class frequency
        if 'train' in txt and not meta:
            if not os.path.exists('cls_freq'):
                os.makedirs('cls_freq')
            freq_path = os.path.join('cls_freq', dataset + '.json')
            self.img_num_per_cls = [0 for _ in range(max(self.labels)+1)]
            for cls in self.labels:
                self.img_num_per_cls[cls] += 1
            with open(freq_path, 'w') as fd:
                json.dump(self.img_num_per_cls, fd)

        path = self.img_path[0]
        label = np.int32(self.labels[0])
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
            sample = sample.numpy()
        
        if txt_split not in os.listdir(root+'/'+dataset):
            if self.xm.is_master_ordinal():
                os.mkdir(root+'/'+dataset+'/'+txt_split)
                self.mmap_samples = self._init_mmap(root+'/'+dataset+'/'+txt_split+'/'+dataset+txt_split+'samples.dat', sample.dtype, (len(self.labels), *sample.shape))
                self.mmap_labels = self._init_mmap(root+'/'+dataset+'/'+txt_split+'/'+dataset+txt_split+'labels.dat', label.dtype, (len(self.labels), *label.shape))
                for i in tqdm(range(len(self.labels)), desc = 'creating memmap'):
                    path = self.img_path[i]
                    label = np.int32(self.labels[i])
                    with open(path, 'rb') as f:
                        sample = Image.open(f).convert('RGB')
                    if self.transform is not None:
                        sample = self.transform(sample)
                        sample = sample.numpy()
                    self.mmap_samples[i] = sample
                    self.mmap_labels[i] = label
        else:
            self.mmap_samples = self._init_mmap(root+'/'+dataset+'/'+txt_split+'/'+dataset+txt_split+'samples.dat', sample.dtype, (len(self.labels), *sample.shape), use_existing=True)
            self.mmap_labels = self._init_mmap(root+'/'+dataset+'/'+txt_split+'/'+dataset+txt_split+'labels.dat', label.dtype, (len(self.labels), *label.shape), use_existing=True)
        self.xm.rendezvous('Creating memmap')

    def __getitem__(self, idx: int):
        sample = torch.tensor(self.mmap_samples[idx])
        label = torch.tensor(self.mmap_labels[idx])
        return sample, label, idx

    def __len__(self) -> int:
        return len(self.labels)

    def _init_mmap(self, path: str, dtype: np.dtype, shape, use_existing = False) -> np.ndarray:
        open_mode = "w+"

        if use_existing:
            open_mode = "r"
        
        return np.memmap(
            path,
            dtype=dtype,
            mode=open_mode,
            shape=shape,
        )


# Load datasets
def load_data(data_root, dataset, phase, batch_size, sampler_dic=None, num_workers=4, test_open=False, shuffle=True, cifar_imb_ratio=None, meta=False):
    if phase == 'train_plain':
        txt_split = 'train'
    elif phase == 'train_val':
        txt_split = 'val'
        phase = 'train'
    else:
        txt_split = phase
    txt = './data/%s/%s_%s.txt'%(dataset, dataset, txt_split)
    # txt = './data/%s/%s_%s.txt'%(dataset, dataset, (phase if phase != 'train_plain' else 'train'))

    print('Loading data from %s' % (txt))


    if dataset == 'iNaturalist18':
        print('===> Loading iNaturalist18 statistics')
        key = 'iNaturalist18'
    else:
        key = 'default'

    if dataset == 'CIFAR10_LT':
        print('====> CIFAR10 Imbalance Ratio: ', cifar_imb_ratio)
        set_ = IMBALANCECIFAR10(phase, imbalance_ratio=cifar_imb_ratio, root=data_root)
    elif dataset == 'CIFAR100_LT':
        print('====> CIFAR100 Imbalance Ratio: ', cifar_imb_ratio)
        set_ = IMBALANCECIFAR100(phase, imbalance_ratio=cifar_imb_ratio, root=data_root)
    else:
        rgb_mean, rgb_std = RGB_statistics[key]['mean'], RGB_statistics[key]['std']
        if phase not in ['train', 'val']:
            transform = get_data_transform('test', rgb_mean, rgb_std, key)
        else:
            transform = get_data_transform(phase, rgb_mean, rgb_std, key)

        print('Use data transformation:', transform)

        # set_ = LT_Dataset(data_root, txt, dataset, transform, meta)
        set_ = MMAPDataset(data_root, txt, dataset, txt_split, transform, meta)

    print(len(set_))

    if sampler_dic and phase == 'train' and sampler_dic.get('batch_sampler', False):
        print('Using sampler: ', sampler_dic['sampler'])
        return DataLoader(dataset=set_,
                           batch_sampler=sampler_dic['sampler'](set_, **sampler_dic['params']),
                           num_workers=num_workers)

    elif sampler_dic and (phase == 'train' or meta):
        print('Using sampler: ', sampler_dic['sampler'])
        # print('Sample %s samples per-class.' % sampler_dic['num_samples_cls'])
        print('Sampler parameters: ', sampler_dic['params'])
        return DataLoader(dataset=set_, batch_size=batch_size, shuffle=False,
                           sampler=sampler_dic['sampler'](set_, **sampler_dic['params']),
                           num_workers=num_workers)
    else:
        print('No sampler.')
        print('Shuffle is %s.' % (shuffle))
        return DataLoader(dataset=set_, batch_size=batch_size,
                          num_workers=num_workers, drop_last=True,
                          pin_memory=True, persistent_workers=True, prefetch_factor=16,
                          sampler = DistributedSampler(set_, xm.xrt_world_size(), xm.get_ordinal(), shuffle=shuffle))
