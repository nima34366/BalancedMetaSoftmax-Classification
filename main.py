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

import os
os.environ['XRT_TPU_CONFIG'] = "localservice;0;localhost:51011"
os.environ['XLA_USE_BF16']                 = '1'
os.environ['XLA_TENSOR_ALLOCATOR_MAXSIZE'] = '1000000000'

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import os
import argparse
import pprint
from data import dataloader
from run_networks import model
import warnings
import yaml
# import tensorflow.compat.v2 as tf2
from utils import source_import, get_value
# import tensorflow as tf

# if __name__ == "__main__":
#     tf2.logging.set_verbosity(tf.logging.INFO)
#     tf2.profiler.experimental.server.start(6000)
#     app.run(main)


data_root = {'ImageNet': '/mnt/disks/persist/imagenet',
             'Places': './dataset/Places-LT',
             'iNaturalist18': '/checkpoint/bykang/iNaturalist18',
             'CIFAR10': './dataset/CIFAR10',
             'CIFAR100': './dataset/CIFAR100',
             }

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default=None, type=str)
parser.add_argument('--test', default=False, action='store_true')
parser.add_argument('--batch_size', type=int, default=None)
parser.add_argument('--test_open', default=False, action='store_true')
parser.add_argument('--output_logits', default=False)
parser.add_argument('--model_dir', type=str, default=None)
parser.add_argument('--save_feat', type=str, default='')

# KNN testing parameters 
parser.add_argument('--knn', default=False, action='store_true')
parser.add_argument('--feat_type', type=str, default='cl2n')
parser.add_argument('--dist_type', type=str, default='l2')

# Learnable tau
parser.add_argument('--val_as_train', default=False, action='store_true')

args = parser.parse_args()

def update(config, args):
    # Change parameters
    config['model_dir'] = get_value(config['model_dir'], args.model_dir)
    config['training_opt']['batch_size'] = \
        get_value(config['training_opt']['batch_size'], args.batch_size)

    # Testing with KNN
    if args.knn and args.test:
        training_opt = config['training_opt']
        classifier_param = {
            'feat_dim': training_opt['feature_dim'],
            'num_classes': training_opt['num_classes'], 
            'feat_type': args.feat_type,
            'dist_type': args.dist_type,
            'log_dir': training_opt['log_dir']}
        classifier = {
            'def_file': './models/KNNClassifier.py',
            'params': classifier_param,
            'optim_params': config['networks']['classifier']['optim_params']}
        config['networks']['classifier'] = classifier
    
    return config

def init_models(config, test_mode, meta_sample):
    networks_defs = config['networks']
    networks = {}


    # xm.master_print("Using", torch.cuda.device_count(), "GPUs.")
    
    for key, val in networks_defs.items():

        # Networks
        def_file = val['def_file']
        # model_args = list(val['params'].values())
        # model_args.append(self.test_mode)
        model_args = val['params']
        model_args.update({'test': test_mode})

        networks[key] = source_import(def_file).create_model(**model_args)
        # print(networks[key])
        if 'KNNClassifier' in type(networks[key]).__name__:
            # Put the KNN classifier on one single GPU
            networks[key] = networks[key] ####
        else:
            # networks[key] = nn.DataParallel(networks[key]).to(device)
            networks[key] = xmp.MpModelWrapper(networks[key])



        # Optimizer list
        
    return networks


# ============================================================================
# LOAD CONFIGURATIONS
with open(args.cfg) as f:
    config = yaml.load(f, Loader=yaml.Loader)
config = update(config, args)

test_mode = args.test
test_open = args.test_open
if test_open:
    test_mode = True
output_logits = args.output_logits
training_opt = config['training_opt']
relatin_opt = config['memory']
dataset = training_opt['dataset']

if not os.path.isdir(training_opt['log_dir']):
    os.makedirs(training_opt['log_dir'])

print('Loading dataset from: %s' % data_root[dataset.rstrip('_LT')])
pprint.pprint(config)

def split2phase(split):
    if split == 'train' and args.val_as_train:
        return 'train_val'
    else:
        return split

sampler_defs = training_opt['sampler']
if sampler_defs and sampler_defs['type'] == 'MetaSampler':
    networks= init_models(config, test_mode, True)
else:
    networks= init_models(config, test_mode, False)


def distrib_train(rank, flags):
    device = xm.xla_device()
    # torch.set_default_tensor_type('torch.FloatTensor')
    os.environ["WORLD_SIZE"] = str(xm.xrt_world_size())
    os.environ["RANK"] = str(xm.get_ordinal())

    if not test_mode:

        sampler_defs = training_opt['sampler']
        if sampler_defs:
            if sampler_defs['type'] == 'ClassAwareSampler':
                sampler_dic = {
                    'sampler': source_import(sampler_defs['def_file']).get_sampler(),
                    'params': {'num_samples_cls': sampler_defs['num_samples_cls']}
                }
            elif sampler_defs['type'] in ['MixedPrioritizedSampler',
                                        'ClassPrioritySampler']:
                sampler_dic = {
                    'sampler': source_import(sampler_defs['def_file']).get_sampler(),
                    'params': {k: v for k, v in sampler_defs.items() \
                            if k not in ['type', 'def_file']}
                }
            elif sampler_defs['type'] == 'MetaSampler':  # Add option for Meta Sampler
                learner = source_import(sampler_defs['def_file']).get_learner()(
                    num_classes=training_opt['num_classes'],
                    init_pow=sampler_defs.get('init_pow', 0.0),
                    freq_path=sampler_defs.get('freq_path', None)
                )
                sampler_dic = {
                    'batch_sampler': True,
                    'sampler': source_import(sampler_defs['def_file']).get_sampler(),
                    'params': {'meta_learner': learner, 'batch_size': training_opt['batch_size']}
                }
        else:
            sampler_dic = None

        splits = ['train', 'train_plain', 'val']
        # if dataset not in ['iNaturalist18', 'ImageNet']:
        if dataset not in ['iNaturalist18', 'ImageNet_LT']:
            splits.append('test')
        data = {x: dataloader.load_data(data_root=data_root[dataset.rstrip('_LT')],
                                        dataset=dataset, phase=split2phase(x), 
                                        batch_size=training_opt['batch_size'],
                                        sampler_dic=sampler_dic,
                                        num_workers=training_opt['num_workers'],
                                        cifar_imb_ratio=training_opt['cifar_imb_ratio'] if 'cifar_imb_ratio' in training_opt else None)
                for x in splits}

        if sampler_defs and sampler_defs['type'] == 'MetaSampler':   # todo: use meta-sampler
            cbs_file = './data/ClassAwareSampler.py'
            cbs_sampler_dic = {
                    'sampler': source_import(cbs_file).get_sampler(),
                    'params': {'is_infinite': True}
                }
            # use Class Balanced Sampler to create meta set
            data['meta'] = dataloader.load_data(data_root=data_root[dataset.rstrip('_LT')],
                                        dataset=dataset, phase='train' if 'CIFAR' in dataset else 'val',
                                        batch_size=sampler_defs.get('meta_batch_size', training_opt['batch_size'], ),
                                        sampler_dic=cbs_sampler_dic,
                                        num_workers=training_opt['num_workers'],
                                        cifar_imb_ratio=training_opt['cifar_imb_ratio'] if 'cifar_imb_ratio' in training_opt else None,
                                        meta=True)
            training_model = model(config, data, device = device, test=False, meta_sample=True, learner=learner, networks=networks)
        else:
            training_model = model(config, data, device = device, test=False, networks=networks)

        training_model.train()

    else:

        warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data",
                                UserWarning)

        print('Under testing phase, we load training data simply to calculate \
            training data number for each class.')

        if 'iNaturalist' in training_opt['dataset']:
            splits = ['train', 'val']
            test_split = 'val'
        else:
            splits = ['train', 'val', 'test']
            test_split = 'test'
        if 'ImageNet' == training_opt['dataset']:
            splits = ['train', 'val']
            test_split = 'val'
        if args.knn or True:
            splits.append('train_plain')

        data = {x: dataloader.load_data(data_root=data_root[dataset.rstrip('_LT')],
                                        dataset=dataset, phase=x,
                                        batch_size=training_opt['batch_size'],
                                        sampler_dic=None, 
                                        test_open=test_open,
                                        num_workers=training_opt['num_workers'],
                                        shuffle=False,
                                        cifar_imb_ratio=training_opt['cifar_imb_ratio'] if 'cifar_imb_ratio' in training_opt else None)
                for x in splits}
        
        training_model = model(config, data, device = device, test=True, networks=networks)
        # training_model.load_model()
        training_model.load_model(args.model_dir)
        if args.save_feat in ['train_plain', 'val', 'test']:
            saveit = True
            test_split = args.save_feat
        else:
            saveit = False
        
        training_model.eval(phase=test_split, openset=test_open, save_feat=saveit)
        
        if output_logits:
            training_model.output_logits(openset=test_open)
        
if __name__ == "__main__":
    xmp.spawn(distrib_train, args=({},), nprocs=8, start_method='fork')

print('ALL COMPLETED.')
