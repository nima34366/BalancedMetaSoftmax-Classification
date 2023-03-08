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
import torch
import torch_xla
import torch_xla.core.xla_model as xm
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Sampler
from torchvision import transforms
import os
from PIL import Image
from data.ImbalanceCIFAR import IMBALANCECIFAR10, IMBALANCECIFAR100
from torch.utils.data import DistributedSampler
from typing import Optional, Iterator, Union
from webdataset import WebDataset, ShardWriter, WebLoader
import numpy as np
from tqdm import tqdm



class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

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

def my_worker_splitter(urls):
   """Split urls per worker
   Selects a subset of urls based on Torch get_worker_info.
   Used as a shard selection function in Dataset.
   replaces split_by_worker"""

   urls = [url for url in urls]

   assert isinstance(urls, list)

   worker_info = torch.utils.data.get_worker_info()
   if worker_info is not None:
       wid = worker_info.id
       num_workers = worker_info.num_workers

       return urls[wid::num_workers]
   else:
       return urls

def my_node_splitter(urls):
   """Split urls_ correctly per accelerator node
   :param urls:
   :return: slice of urls_
   """
   rank=xm.get_ordinal()
   num_replicas=xm.xrt_world_size()

   urls_this = urls[rank::num_replicas]
  
   return urls_this

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

class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.
    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.
        Args:
            index: index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)

class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
                distributed training
            rank (int, optional): Rank of the current process
                within ``num_replicas``
            shuffle (bool, optional): If true (default),
                sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        """Iterate over sampler.
        Returns:
            python iterator
        """
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))


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
            # sample = f.read()

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
        
        if self.xm.is_master_ordinal():
            if txt_split not in os.listdir(root+'/'+dataset):
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

    # print()
    
    # if xm.is_master_ordinal():
    #     if txt_split not in os.listdir(data_root+'/'+dataset):
    #         os.mkdir(data_root+'/'+dataset+'/'+txt_split)
    #         with ShardWriter(data_root+'/'+dataset+'/'+txt_split+'/'+dataset+txt_split+"-%06d.tar", maxcount=10000) as sink:
    #             for sample, label, index in set_:
    #                 if index%1000==0: print(index)
    #                 sink.write({
    #                     "__key__": str(index),
    #                     "sample.jpeg": sample,
    #                     "label.cls": label,
    #                     "index.cls": index
    #                 })
    # xm.rendezvous('Creating webdataset')
    # dataset_size = len(set_)
    # def dataset_len(self):
    #     return dataset_size
    # WebDataset.__len__ = dataset_len
    # set_ = WebDataset(data_root+'/'+dataset+'/'+txt_split+'/'+dataset+txt_split+'-{000000..0000'+str(len(os.listdir(data_root+'/'+dataset+'/'+txt_split))-1)+'}.tar').decode("pil").to_tuple("sample.jpeg","label.cls","index.cls").map_tuple(transform, lambda x:torch.tensor(x), lambda x:torch.tensor(x))
    # if (shuffle):
    #     set_ = WebDataset(data_root+'/'+dataset+'/'+txt_split+'/'+dataset+txt_split+'-{000000..0000'+str(len(os.listdir(data_root+'/'+dataset+'/'+txt_split))-1)+'}.tar', resampled=True).shuffle(1000).decode("pil").to_tuple("sample.jpeg","label.cls","index.cls").map_tuple(transform, lambda x:torch.tensor(x), lambda x:torch.tensor(x)).batched(batch_size, partial=False).with_epoch(dataset_size//batch_size)
    # else:
    #     set_ = WebDataset(data_root+'/'+dataset+'/'+txt_split+'/'+dataset+txt_split+'-{000000..0000'+str(len(os.listdir(data_root+'/'+dataset+'/'+txt_split))-1)+'}.tar', resampled=True).decode("pil").to_tuple("sample.jpeg","label.cls","index.cls").map_tuple(transform, lambda x:torch.tensor(x), lambda x:torch.tensor(x)).batched(batch_size, partial=False).with_epoch(dataset_size//batch_size)



    if sampler_dic and phase == 'train' and sampler_dic.get('batch_sampler', False):
        print('Using sampler: ', sampler_dic['sampler'])
        return DataLoader(dataset=set_,
                           batch_sampler=DistributedSamplerWrapper(sampler_dic['sampler'](set_, **sampler_dic['params']), xm.xrt_world_size(), xm.get_ordinal(), shuffle=False),
                           num_workers=num_workers, drop_last=False)

    elif sampler_dic and (phase == 'train' or meta):
        print('Using sampler: ', sampler_dic['sampler'])
        # print('Sample %s samples per-class.' % sampler_dic['num_samples_cls'])
        print('Sampler parameters: ', sampler_dic['params'])
        return DataLoader(dataset=set_, batch_size=None, shuffle=False, drop_last=False,
                           sampler=DistributedSamplerWrapper(sampler_dic['sampler'](set_, **sampler_dic['params']), xm.xrt_world_size(), xm.get_ordinal(), shuffle=False),
                           num_workers=num_workers)
    else:
        print('No sampler.')
        print('Shuffle is %s.' % (shuffle))
        # loader = WebLoader(dataset=set_, num_workers=num_workers, batch_size=None)
        # if shuffle:
        #     return loader.unbatched().shuffle(1000).batched(batch_size)
        # else:
        #     return loader.unbatched().batched(batch_size)
        # return DataLoader(dataset=set_, batch_size=None, num_workers=num_workers, prefetch_factor=8)
        # return set_
        # return DataLoader(pin_memory=True, persistent_workers=True, dataset=set_, batch_size=batch_size, num_workers=num_workers, prefetch_factor=32, sampler = DistributedSampler(set_, xm.xrt_world_size(), xm.get_ordinal(), shuffle=shuffle))
        # return DataLoader(pin_memory=True, persistent_workers=True, dataset=set_, batch_size=batch_size, num_workers=num_workers, prefetch_factor=16)
        return DataLoader(pin_memory=True, persistent_workers=True, dataset=set_, batch_size=batch_size, num_workers=num_workers, prefetch_factor=16, drop_last=True, sampler = DistributedSampler(set_, xm.xrt_world_size(), xm.get_ordinal(), shuffle=False))


