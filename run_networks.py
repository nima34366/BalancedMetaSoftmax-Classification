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
import copy
import pickle
import torch
# import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from utils import *
from logger import Logger
import time
import numpy as np
import warnings
import pdb
import time
import gc
import higher
from itertools import islice
# import torch.profiler
# import torch_xla.debug.profiler as xp


class model ():
    
    def __init__(self, config, data, device, networks, test=False, meta_sample=False, learner=None):

        self.meta_sample = meta_sample

        # init meta learner and meta set
        if self.meta_sample:
            assert learner is not None
            self.learner = learner
            self.meta_data = iter(data['meta'])
        
        # self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.config = config
        self.training_opt = self.config['training_opt']
        self.memory = self.config['memory']
        self.data = data
        self.test_mode = test
        self.num_gpus = torch.cuda.device_count()
        self.do_shuffle = config['shuffle'] if 'shuffle' in config else False
        

        # Compute epochs from iterations
        if self.training_opt.get('num_iterations', False):
            self.training_opt['num_epochs'] = math.ceil(self.training_opt['num_iterations'] / len(self.data['train']))
        if self.config.get('warmup_iterations', False):
            self.config['warmup_epochs'] = math.ceil(self.config['warmup_iterations'] / len(self.data['train']))

        # Setup logger
        self.logger = Logger(self.training_opt['log_dir'])
        
        # Initialize model
        self.networks = networks
        self.init_params()
        # self.init_models()
        # self.model_optim_params_list = model_optim_params_list
        # Load pre-trained model parameters
        if 'model_dir' in self.config and self.config['model_dir'] is not None:
            self.load_model(self.config['model_dir'])

        # Under training mode, initialize training steps, optimizers, schedulers, criterions, and centroids
        if not self.test_mode:

            # If using steps for training, we need to calculate training steps 
            # for each epoch based on actual number of training data instead of 
            # oversampled data number 
            xm.master_print('Using steps for training.')
            self.training_data_num = len(self.data['train'].dataset)
            # xm.master_print(self.training_data_num, self.training_opt['batch_size'])
            self.epoch_steps = int(self.training_data_num/self.training_opt['batch_size'])
            xm.master_print('Num epoch steps', self.epoch_steps)

            # Initialize model optimizer and scheduler
            xm.master_print('Initializing model optimizer.')
            self.scheduler_params = self.training_opt['scheduler_params']
            self.model_optimizer, \
            self.model_optimizer_scheduler = self.init_optimizers(self.model_optim_params_list)
            self.init_criterions()
            if self.memory['init_centroids']:
                self.criterions['FeatureLoss'].centroids.data = \
                    self.centroids_cal(self.data['train_plain'])
            
            # Set up log file
            self.log_file = os.path.join(self.training_opt['log_dir'], 'log.txt')
            if os.path.isfile(self.log_file):
                os.remove(self.log_file)
            self.logger.log_cfg(self.config)
        else:
            if 'KNNClassifier' in self.config['networks']['classifier']['def_file']:
                self.load_model()
                if not self.networks['classifier'].initialized:
                    cfeats = self.get_knncentroids()
                    xm.master_print('===> Saving features to %s' % 
                          os.path.join(self.training_opt['log_dir'], 'cfeats.pkl'))
                    with open(os.path.join(self.training_opt['log_dir'], 'cfeats.pkl'), 'wb') as f:
                        pickle.dump(cfeats, f)
                    self.networks['classifier'].update(cfeats)
            self.log_file = None
        
    def init_params(self):
        self.model_optim_params_list = []
        networks_defs = self.config['networks']
        for key, val in networks_defs.items():

            self.networks[key] = self.networks[key].to(self.device)

            if 'fix' in val and val['fix']:
                xm.master_print('Freezing feature weights except for self attention weights (if exist).')
            for param_name, param in self.networks[key].named_parameters():
                # Freeze all parameters except self attention parameters
                if 'selfatt' not in param_name and 'fc' not in param_name:
                    param.requires_grad = False
            # xm.master_print('  | ', param_name, param.requires_grad)

            if self.meta_sample and key!='classifier':
                # avoid adding classifier parameters to the optimizer,
                # otherwise error will be raised when computing higher gradients
                continue

            optim_params = val['optim_params']
            self.model_optim_params_list.append({'params': self.networks[key].parameters(),
                                                'lr': optim_params['lr'],
                                                'momentum': optim_params['momentum'],
                                                'weight_decay': optim_params['weight_decay']})

    def init_criterions(self):
        criterion_defs = self.config['criterions']
        self.criterions = {}
        self.criterion_weights = {}

        for key, val in criterion_defs.items():
            def_file = val['def_file']
            loss_args = list(val['loss_params'].values())

            self.criterions[key] = source_import(def_file).create_loss(*loss_args).to(self.device)
            self.criterion_weights[key] = val['weight']
          
            if val['optim_params']:
                xm.master_print('Initializing criterion optimizer.')
                optim_params = val['optim_params']
                optim_params = [{'params': self.criterions[key].parameters(),
                                'lr': optim_params['lr'],
                                'momentum': optim_params['momentum'],
                                'weight_decay': optim_params['weight_decay']}]
                # Initialize criterion optimizer and scheduler
                self.criterion_optimizer, \
                self.criterion_optimizer_scheduler = self.init_optimizers(optim_params)
            else:
                self.criterion_optimizer = None

    def init_optimizers(self, optim_params):
        optimizer = optim.SGD(optim_params)

        if self.meta_sample:
            # init meta optimizer
            self.optimizer_meta = torch.optim.Adam(self.learner.parameters(),
                                                   lr=self.training_opt['sampler'].get('lr', 0.01))
        if self.config['coslr']:
            xm.master_print("===> Using coslr eta_min={}".format(self.config['endlr']))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, self.training_opt['num_epochs'], eta_min=self.config['endlr'])
        elif self.config['coslrwarmup']:
            xm.master_print("===> Using coslrwarmup eta_min={}, warmup_epochs={}".format(
                self.config['endlr'],self.config['warmup_epochs']))
            scheduler = CosineAnnealingLRWarmup(
                optimizer=optimizer,
                T_max=self.training_opt['num_epochs'],
                eta_min=self.config['endlr'],
                warmup_epochs=self.config['warmup_epochs'],
                base_lr=self.config['base_lr'],
                warmup_lr=self.config['warmup_lr']
            )
        else:
            scheduler = optim.lr_scheduler.StepLR(optimizer,
                                                  step_size=self.scheduler_params['step_size'],
                                                  gamma=self.scheduler_params['gamma'])
        return optimizer, scheduler

    def batch_forward (self, inputs, labels=None, centroids=False, feature_ext=False, phase='train'):
        '''
        This is a general single batch running function. 
        '''
        self.model_optimizer.zero_grad()
        if self.criterion_optimizer:
            self.criterion_optimizer.zero_grad()

        # Calculate Features
        self.features, self.feature_maps = self.networks['feat_model'](inputs)

        # If not just extracting features, calculate logits
        if not feature_ext:

            # During training, calculate centroids if needed to 
            if phase != 'test':
                if centroids and 'FeatureLoss' in self.criterions.keys():
                    self.centroids = self.criterions['FeatureLoss'].centroids.data
                    torch.cat([self.centroids] * self.num_gpus)
                else:
                    self.centroids = None

            if self.centroids is not None:
                centroids_ = torch.cat([self.centroids] * self.num_gpus)
            else:
                centroids_ = self.centroids

            # Calculate logits with classifier
            self.logits, self.direct_memory_feature = self.networks['classifier'](self.features, centroids_)

    def batch_backward(self):
        # Zero out optimizer gradients
        # Back-propagation from loss outputs
        self.loss.backward()
        # Step optimizers
        # self.model_optimizer.step()
        xm.optimizer_step(self.model_optimizer, barrier=False)
        # xm.mark_step()
        if self.criterion_optimizer:
            xm.optimizer_step(self.criterion_optimizer, barrier=False)

    def batch_loss(self, labels):
        self.loss = 0

        # First, apply performance loss
        if 'PerformanceLoss' in self.criterions.keys():
            self.loss_perf = self.criterions['PerformanceLoss'](self.logits, labels)
            self.loss_perf *=  self.criterion_weights['PerformanceLoss']
            self.loss += self.loss_perf

        # Apply loss on features if set up
        if 'FeatureLoss' in self.criterions.keys():
            self.loss_feat = self.criterions['FeatureLoss'](self.features, labels)
            self.loss_feat = self.loss_feat * self.criterion_weights['FeatureLoss']
            # Add feature loss to total loss
            self.loss += self.loss_feat
    
    def shuffle_batch(self, x, y):
        index = torch.randperm(x.size(0))
        x = x[index]
        y = y[index]
        return x, y

    def meta_forward(self, inputs, labels, verbose=False):
        # take a meta step in the inner loop
        self.learner.train()
        self.model_optimizer.zero_grad()
        self.optimizer_meta.zero_grad()
        with higher.innerloop_ctx(self.networks['classifier'], self.model_optimizer) as (fmodel, diffopt):
            # obtain the surrogate model
            features, _ = self.networks['feat_model'](inputs)
            train_outputs, _ = fmodel(features.detach())
            loss = self.criterions['PerformanceLoss'](train_outputs, labels, reduction='none')
            loss = self.learner.forward_loss(loss)
            diffopt.step(loss)

            # use the surrogate model to update sample rate
            val_inputs, val_targets, _ = next(self.meta_data)
            val_inputs = val_inputs.to(self.device)
            val_targets = val_targets.to(self.device)
            features, _ = self.networks['feat_model'](val_inputs)
            val_outputs, _ = fmodel(features.detach())
            val_loss = F.cross_entropy(val_outputs, val_targets, reduction='mean')
            val_loss.backward()
            self.optimizer_meta.step()

        self.learner.eval()

        if verbose:
            # log the sample rates
            num_classes = self.learner.num_classes
            prob = self.learner.fc[0].weight.sigmoid().squeeze(0)
            print_str = ['Unnormalized Sample Prob:']
            interval = 1 if num_classes < 10 else num_classes // 10
            for i in range(0, num_classes, interval):
                print_str.append('class{}={:.3f},'.format(i, prob[i].item()))
            # max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
            # print_str.append('\nMax Mem: {:.0f}M'.format(max_mem_mb))
            print_write(print_str, self.log_file)

    def train(self):
        # When training the network
        # server = xp.start_server(9012)
        # print_str = ['Phase: train']
        # print_write(print_str, self.log_file)
        # time.sleep(0.25)

        # print_write(['Do shuffle??? --- ', self.do_shuffle], self.log_file)

        # Initialize best model
        # best_model_weights = {}
        # best_model_weights['feat_model'] = copy.deepcopy(self.networks['feat_model'].state_dict())
        # best_model_weights['classifier'] = copy.deepcopy(self.networks['classifier'].state_dict())
        # best_acc = 0.0
        # best_epoch = 0
        # best_centroids = self.centroids

        end_epoch = self.training_opt['num_epochs']

        # Loop over epochs
        xm.rendezvous('init')
        start = time.time()
        # para_loader = self.data['train']
        for epoch in range(1, end_epoch + 1):
            xm.master_print('Time for epoch',epoch,time.time() - start)
            start = time.time()
            xm.master_print(1)
            para_loader = pl.ParallelLoader(self.data['train'], [self.device])
            # self.data['train'].sampler.set_epoch(epoch)
            xm.master_print(2)
            for model in self.networks.values():
                model.train()
            xm.master_print(3)

            # torch.cuda.empty_cache()
            # gc.collect()
            # Set model modes and set scheduler
            # In training, step optimizer scheduler and set model to train() 
            xm.master_print(6)
            # Iterate over dataset
            # total_preds = []
            # total_labels = []
            xm.master_print(8)
            # prof = torch.profiler.profile(
            #     schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            #     on_trace_ready=torch.profiler.tensorboard_trace_handler('/home/wickramasinghenlssck/BalancedMetaSoftmax-Classification/logs/tens'),
            #     record_shapes=True,
            #     with_stack=True)
            # prof.start()
            xm.master_print(self.epoch_steps)
            for step, (inputs, labels, indexes) in enumerate(para_loader.per_device_loader(self.device)):
            # for step, (inputs, labels, indexes) in enumerate(para_loader):
                # Break when step equal to epoch step
                # with xp.StepTrace('train_loop', step_num=step):
                xm.master_print(9)
                # if step == self.epoch_steps:
                #     break
                # if self.do_shuffle:
                #     inputs, labels = self.shuffle_batch(inputs, labels)
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                xm.master_print(10)
                # If on training phase, enable gradients
                # with torch.set_grad_enabled(True):
                if self.meta_sample:
                    # do inner loop
                    self.meta_forward(inputs, labels, verbose=step % self.training_opt['display_step'] == 0)
                xm.master_print(11)
                # If training, forward with loss, and no top 5 accuracy calculation
                self.batch_forward(inputs, labels, 
                                centroids=self.memory['centroids'],
                                phase='train')
                xm.master_print(12)
                self.batch_loss(labels)
                self.batch_backward()
                xm.master_print(13)
                # Tracking predictions
                # _, preds = torch.max(self.logits, 1)
                # total_preds.append(torch2numpy(preds))
                # total_labels.append(torch2numpy(labels))
                xm.master_print(14)
                # Output minibatch training results
                # if step % self.training_opt['display_step'] == 0:

                #     minibatch_loss_feat = self.loss_feat.item() \
                #         if 'FeatureLoss' in self.criterions.keys() else None
                #     minibatch_loss_perf = self.loss_perf.item() \
                #         if 'PerformanceLoss' in self.criterions else None
                #     minibatch_loss_total = self.loss.item()
                #     minibatch_acc = mic_acc_cal(preds, labels)

                #     print_str = ['Epoch: [%d/%d]' 
                #                 % (epoch, self.training_opt['num_epochs']),
                #                 'Step: %5d' 
                #                 % (step),
                #                 'Minibatch_loss_feature: %.3f' 
                #                 % (minibatch_loss_feat) if minibatch_loss_feat else '',
                #                 'Minibatch_loss_performance: %.3f'
                #                 % (minibatch_loss_perf) if minibatch_loss_perf else '',
                #                 'Minibatch_accuracy_micro: %.3f'
                #                 % (minibatch_acc)]
                #     print_write(print_str, self.log_file)

                #     loss_info = {
                #         'Epoch': epoch,
                #         'Step': step,
                #         'Total': minibatch_loss_total,
                #         'CE': minibatch_loss_perf,
                #         'feat': minibatch_loss_feat
                #     }

                # del inputs,labels
                # del inputs,labels, self.logits, self.direct_memory_feature, self.centroids, self.features, self.feature_maps, self.loss, self.loss_perf

                # gc.collect()
                xm.master_print(15)
                            # xm.master_print(met.metrics_report())
                            # self.logger.log_loss(loss_info)

                    # Update priority weights if using PrioritizedSampler
                    # if self.training_opt['sampler'] and \
                    #    self.training_opt['sampler']['type'] == 'PrioritizedSampler':
                    # if hasattr(self.data['train'].sampler, 'update_weights'):
                    #     if hasattr(self.data['train'].sampler, 'ptype'):
                    #         ptype = self.data['train'].sampler.ptype 
                    #     else:
                    #         ptype = 'score'
                    #     ws = get_priority(ptype, self.logits.detach(), labels)
                    #     # ws = logits2score(self.logits.detach(), labels)
                    #     inlist = [indexes.cpu().numpy(), ws]
                    #     if self.training_opt['sampler']['type'] == 'ClassPrioritySampler':
                    #         inlist.append(labels.cpu().numpy())
                    #     self.data['train'].sampler.update_weights(*inlist)
                        # self.data['train'].sampler.update_weights(indexes.cpu().numpy(), ws)
                xm.rendezvous('step')
            # prof.stop()
            # if hasattr(self.data['train'].sampler, 'get_weights'):
            #     self.logger.log_ws(epoch, self.data['train'].sampler.get_weights())
            # if hasattr(self.data['train'].sampler, 'reset_weights'):
            #     self.data['train'].sampler.reset_weights(epoch)
            self.model_optimizer_scheduler.step()
            xm.master_print(5)
            if self.criterion_optimizer:
                self.criterion_optimizer_scheduler.step()
            # del indexes
            # gc.collect()
            # After every epoch, validation
            xm.master_print(16)
            # rsls = {'epoch': epoch}
            xm.master_print(17)
            # rsls_train = self.eval_with_preds(total_preds, total_labels)
            # rsls_eval = self.eval(phase='val')
            xm.master_print(18)
            # rsls.update(rsls_train)
            # rsls.update(rsls_eval)

            # Reset class weights for sampling if pri_mode is valid
            # if hasattr(self.data['train'].sampler, 'reset_priority'):
            #     ws = get_priority(self.data['train'].sampler.ptype,
            #                       self.total_logits.detach(),
            #                       self.total_labels)
            #     self.data['train'].sampler.reset_priority(ws, self.total_labels.cpu().numpy())

            # Log results
            # self.logger.log_acc(rsls)

            # Under validation, the best model need to be updated
            # if self.eval_acc_mic_top1 > best_acc:
            #     best_epoch = epoch
            #     best_acc = self.eval_acc_mic_top1
            #     best_centroids = self.centroids
            #     best_model_weights['feat_model'] = copy.deepcopy(self.networks['feat_model'].state_dict())
            #     best_model_weights['classifier'] = copy.deepcopy(self.networks['classifier'].state_dict())
            
            # xm.master_print('===> Saving checkpoint')
            # self.save_latest(epoch)
            xm.rendezvous('all_collected')

        xm.master_print()
        xm.master_print('Training Complete.')

        print_str = ['Best validation accuracy is %.3f at epoch %d' % (best_acc, best_epoch)]
        print_write(print_str, self.log_file)
        # Save the best model and best centroids if calculated
        self.save_model(epoch, best_epoch, best_model_weights, best_acc, centroids=best_centroids)

        # Test on the test set
        self.reset_model(best_model_weights)
        self.eval('test' if 'test' in self.data else 'val')
        xm.master_print('Done')
    
    def eval_with_preds(self, preds, labels):
        # Count the number of examples
        n_total = sum([len(p) for p in preds])

        # Split the examples into normal and mixup
        normal_preds, normal_labels = [], []
        mixup_preds, mixup_labels1, mixup_labels2, mixup_ws = [], [], [], []
        for p, l in zip(preds, labels):
            if isinstance(l, tuple):
                mixup_preds.append(p)
                mixup_labels1.append(l[0])
                mixup_labels2.append(l[1])
                mixup_ws.append(l[2] * np.ones_like(l[0]))
            else:
                normal_preds.append(p)
                normal_labels.append(l)
        
        # Calculate normal prediction accuracy
        rsl = {'train_all':0., 'train_many':0., 'train_median':0., 'train_low': 0.}
        if len(normal_preds) > 0:
            normal_preds, normal_labels = list(map(np.concatenate, [normal_preds, normal_labels]))
            n_top1 = mic_acc_cal(normal_preds, normal_labels)
            n_top1_many, \
            n_top1_median, \
            n_top1_low, = shot_acc(normal_preds, normal_labels, self.data['train'])
            rsl['train_all'] += len(normal_preds) / n_total * n_top1
            rsl['train_many'] += len(normal_preds) / n_total * n_top1_many
            rsl['train_median'] += len(normal_preds) / n_total * n_top1_median
            rsl['train_low'] += len(normal_preds) / n_total * n_top1_low

        # Calculate mixup prediction accuracy
        if len(mixup_preds) > 0:
            mixup_preds, mixup_labels, mixup_ws = \
                list(map(np.concatenate, [mixup_preds*2, mixup_labels1+mixup_labels2, mixup_ws]))
            mixup_ws = np.concatenate([mixup_ws, 1-mixup_ws])
            n_top1 = weighted_mic_acc_cal(mixup_preds, mixup_labels, mixup_ws)
            n_top1_many, \
            n_top1_median, \
            n_top1_low, = weighted_shot_acc(mixup_preds, mixup_labels, mixup_ws, self.data['train'])
            rsl['train_all'] += len(mixup_preds) / 2 / n_total * n_top1
            rsl['train_many'] += len(mixup_preds) / 2 / n_total * n_top1_many
            rsl['train_median'] += len(mixup_preds) / 2 / n_total * n_top1_median
            rsl['train_low'] += len(mixup_preds) / 2 / n_total * n_top1_low

        # Top-1 accuracy and additional string
        print_str = ['\n Training acc Top1: %.3f \n' % (rsl['train_all']),
                     'Many_top1: %.3f' % (rsl['train_many']),
                     'Median_top1: %.3f' % (rsl['train_median']),
                     'Low_top1: %.3f' % (rsl['train_low']),
                     '\n']
        print_write(print_str, self.log_file)

        return rsl

    def eval(self, phase='val', openset=False, save_feat=False):
        start = time.process_time()
        print_str = ['Phase: %s' % (phase)]
        # print_write(print_str, self.log_file)
        # time.sleep(0.25)
        xm.master_print(19,time.process_time() - start)
        start = time.process_time()
        if openset:
            xm.master_print('Under openset test mode. Open threshold is %.1f' 
                  % self.training_opt['open_threshold'])
        xm.master_print(20,time.process_time() - start)
        start = time.process_time()
        # torch.cuda.empty_cache()
        gc.collect()

        # In validation or testing mode, set model to eval() and initialize running loss/correct
        for model in self.networks.values():
            model.to(self.device).eval()
        xm.master_print(21,time.process_time() - start)
        start = time.process_time()
        self.total_logits = torch.empty((0, self.training_opt['num_classes'])).to(self.device)
        xm.master_print(22,time.process_time() - start)
        start = time.process_time()
        self.total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        xm.master_print(23,time.process_time() - start)
        start = time.process_time()
        self.total_paths = np.empty(0)
        xm.master_print(24,time.process_time() - start)
        start = time.process_time()
        get_feat_only = save_feat
        xm.master_print(25,time.process_time() - start)
        start = time.process_time()
        feats_all, labels_all, idxs_all, logits_all = [], [], [], []
        xm.master_print(26,time.process_time() - start)
        start = time.process_time()
        featmaps_all = []
        xm.master_print(27,time.process_time() - start)
        start = time.process_time()
        # Iterate over dataset
        para_loader = pl.ParallelLoader(self.data[phase], [self.device])
        xm.master_print(28,time.process_time() - start)
        start = time.process_time()
        for inputs, labels, paths in para_loader.per_device_loader(self.device):
            xm.master_print(29,time.process_time() - start)
            start = time.process_time()
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            xm.master_print(30,time.process_time() - start)
            start = time.process_time()
            # If on training phase, enable gradients
            with torch.set_grad_enabled(False):
                xm.master_print(31,time.process_time() - start)
                start = time.process_time()
                # In validation or testing
                self.batch_forward(inputs, labels, 
                                   centroids=self.memory['centroids'],
                                   phase=phase)
                xm.master_print(32,time.process_time() - start)
                start = time.process_time()
                if not get_feat_only:
                    self.total_logits = torch.cat((self.total_logits, self.logits))
                    self.total_labels = torch.cat((self.total_labels, labels))
                    self.total_paths = np.concatenate((self.total_paths, paths.cpu().numpy()))
                xm.master_print(33,time.process_time() - start)
                start = time.process_time()
                if get_feat_only:
                    logits_all.append(self.logits.cpu().numpy())
                    feats_all.append(self.features.cpu().numpy())
                    labels_all.append(labels.cpu().numpy())
                    idxs_all.append(paths.numpy())
                xm.master_print(34,time.process_time() - start)
                start = time.process_time()
                del inputs, labels
                gc.collect()
            xm.master_print(35,time.process_time() - start)
        # del para_loader
        gc.collect()
        if get_feat_only:
            typ = 'feat'
            if phase == 'train_plain':
                name = 'train{}_all.pkl'.format(typ)
            elif phase == 'test':
                name = 'test{}_all.pkl'.format(typ)
            elif phase == 'val':
                name = 'val{}_all.pkl'.format(typ)

            fname = os.path.join(self.training_opt['log_dir'], name)
            xm.master_print('===> Saving feats to ' + fname)
            with open(fname, 'wb') as f:
                pickle.dump({
                             'feats': np.concatenate(feats_all),
                             'labels': np.concatenate(labels_all),
                             'idxs': np.concatenate(idxs_all),
                            },
                            f, protocol=4) 
            return 
        xm.master_print(35,time.process_time() - start)
        start = time.process_time()
        probs, preds = F.softmax(self.total_logits.detach(), dim=1).max(dim=1)

        if openset:
            preds[probs < self.training_opt['open_threshold']] = -1
            self.openset_acc = mic_acc_cal(preds[self.total_labels == -1],
                                            self.total_labels[self.total_labels == -1])
            xm.master_print('\n\nOpenset Accuracy: %.3f' % self.openset_acc)

        # Calculate the overall accuracy and F measurement
        xm.master_print(36,time.process_time() - start)
        start = time.process_time()
        self.eval_acc_mic_top1= mic_acc_cal(preds[self.total_labels != -1],
                                            self.total_labels[self.total_labels != -1])
        self.eval_f_measure = F_measure(preds, self.total_labels, openset=openset,
                                        theta=self.training_opt['open_threshold'])
        # self.many_acc_top1, \
        # self.median_acc_top1, \
        # self.low_acc_top1, \
        # self.cls_accs = shot_acc(preds[self.total_labels != -1],
        #                          self.total_labels[self.total_labels != -1], 
        #                          self.data['train'],
        #                          acc_per_cls=True)
        xm.master_print(37,time.process_time() - start)
        start = time.process_time()
        # Top-1 accuracy and additional string
        # print_str = ['\n\n',
        #              'Phase: %s' 
        #              % (phase),
        #              '\n\n',
        #              'Evaluation_accuracy_micro_top1: %.3f' 
        #              % (self.eval_acc_mic_top1),
        #              '\n',
        #              'Averaged F-measure: %.3f' 
        #              % (self.eval_f_measure),
                    #  '\n',
                    #  'Many_shot_accuracy_top1: %.3f' 
                    #  % (self.many_acc_top1),
                    #  'Median_shot_accuracy_top1: %.3f' 
                    #  % (self.median_acc_top1),
                    #  'Low_shot_accuracy_top1: %.3f' 
                    #  % (self.low_acc_top1),
                    #  '\n']
        
        # rsl = {phase + '_all': self.eval_acc_mic_top1,
        #        phase + '_many': self.many_acc_top1,
        #        phase + '_median': self.median_acc_top1,
        #        phase + '_low': self.low_acc_top1,
        #        phase + '_fscore': self.eval_f_measure}

        # if phase == 'val':
        #     xm.master_print(print_str)
        #     xm.master_print(38,time.process_time() - start)
        #     start = time.process_time()
        # else:
        #     acc_str = ["{:.1f} \t {:.1f} \t {:.1f} \t {:.1f}".format(
        #         self.many_acc_top1 * 100,
        #         self.median_acc_top1 * 100,
        #         self.low_acc_top1 * 100,
        #         self.eval_acc_mic_top1 * 100)]
        #     if self.log_file is not None and os.path.exists(self.log_file):
        #         print_write(print_str, self.log_file)
        #         print_write(acc_str, self.log_file)
        #     else:
        #         xm.master_print(*print_str)
        #         xm.master_print(*acc_str)
        
        # if phase == 'test':
        #     with open(os.path.join(self.training_opt['log_dir'], 'cls_accs.pkl'), 'wb') as f:
        #         pickle.dump(self.cls_accs, f)
        # return rsl
            
    def centroids_cal(self, data, save_all=False):

        centroids = torch.zeros(self.training_opt['num_classes'],
                                   self.training_opt['feature_dim']).to(self.device)

        xm.master_print('Calculating centroids.')

        # torch.cuda.empty_cache()
        gc.collect()
        for model in self.networks.values():
            model.eval()

        feats_all, labels_all, idxs_all = [], [], []

        # Calculate initial centroids only on training data.
        with torch.set_grad_enabled(False):
            for inputs, labels, idxs in tqdm(data):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Calculate Features of each training data
                self.batch_forward(inputs, feature_ext=True)
                # Add all calculated features to center tensor
                for i in range(len(labels)):
                    label = labels[i]
                    centroids[label] += self.features[i]
                # Save features if requried
                if save_all:
                    feats_all.append(self.features.cpu().numpy())
                    labels_all.append(labels.cpu().numpy())
                    idxs_all.append(idxs.numpy())
        
        if save_all:
            fname = os.path.join(self.training_opt['log_dir'], 'feats_all.pkl')
            with open(fname, 'wb') as f:
                pickle.dump({'feats': np.concatenate(feats_all),
                             'labels': np.concatenate(labels_all),
                             'idxs': np.concatenate(idxs_all)},
                            f)
        # Average summed features with class count
        centroids /= torch.tensor(class_count(data)).float().unsqueeze(1).to(self.device)

        return centroids

    def get_knncentroids(self):
        datakey = 'train_plain'
        assert datakey in self.data

        xm.master_print('===> Calculating KNN centroids.')

        # torch.cuda.empty_cache()
        gc.collect()
        for model in self.networks.values():
            model.eval()

        feats_all, labels_all = [], []

        # Calculate initial centroids only on training data.
        with torch.set_grad_enabled(False):
            for inputs, labels, idxs in tqdm(self.data[datakey]):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Calculate Features of each training data
                self.batch_forward(inputs, feature_ext=True)

                feats_all.append(self.features.cpu().numpy())
                labels_all.append(labels.cpu().numpy())
        
        feats = np.concatenate(feats_all)
        labels = np.concatenate(labels_all)

        featmean = feats.mean(axis=0)

        def get_centroids(feats_, labels_):
            centroids = []        
            for i in np.unique(labels_):
                centroids.append(np.mean(feats_[labels_==i], axis=0))
            return np.stack(centroids)
        # Get unnormalized centorids
        un_centers = get_centroids(feats, labels)
    
        # Get l2n centorids
        l2n_feats = torch.Tensor(feats.copy())
        norm_l2n = torch.norm(l2n_feats, 2, 1, keepdim=True)
        l2n_feats = l2n_feats / norm_l2n
        l2n_centers = get_centroids(l2n_feats.numpy(), labels)

        # Get cl2n centorids
        cl2n_feats = torch.Tensor(feats.copy())
        cl2n_feats = cl2n_feats - torch.Tensor(featmean)
        norm_cl2n = torch.norm(cl2n_feats, 2, 1, keepdim=True)
        cl2n_feats = cl2n_feats / norm_cl2n
        cl2n_centers = get_centroids(cl2n_feats.numpy(), labels)

        return {'mean': featmean,
                'uncs': un_centers,
                'l2ncs': l2n_centers,   
                'cl2ncs': cl2n_centers}
    
    def reset_model(self, model_state):
        for key, model in self.networks.items():
            weights = model_state[key]
            weights = {k: weights[k] for k in weights if k in model.state_dict()}
            model.load_state_dict(weights)

    def load_model(self, model_dir=None):
        model_dir = self.training_opt['log_dir'] if model_dir is None else model_dir
        if not model_dir.endswith('.pth'):
            model_dir = os.path.join(model_dir, 'final_model_checkpoint.pth')
        
        xm.master_print('Validation on the best model.')
        xm.master_print('Loading model from %s' % (model_dir))
        
        checkpoint = torch.load(model_dir)          
        model_state = checkpoint['state_dict_best']
        
        self.centroids = checkpoint['centroids'] if 'centroids' in checkpoint else None
        
        for key, model in self.networks.items():
            # if not self.test_mode and key == 'classifier':
            if not self.test_mode and \
                'DotProductClassifier' in self.config['networks'][key]['def_file']:
                # Skip classifier initialization 
                xm.master_print('Skiping classifier initialization')
                continue
            weights = model_state[key]
            weights = {k: weights[k] for k in weights if k in model.state_dict()}
            x = model.state_dict()
            x.update(weights)
            model.load_state_dict(x)
    
    def save_latest(self, epoch):
        model_weights = {}
        model_weights['feat_model'] = copy.deepcopy(self.networks['feat_model'].state_dict())
        model_weights['classifier'] = copy.deepcopy(self.networks['classifier'].state_dict())

        model_states = {
            'epoch': epoch,
            'state_dict': model_weights
        }

        model_dir = os.path.join(self.training_opt['log_dir'], 
                                 'latest_model_checkpoint.pth')
        xm.save(model_states, model_dir)
        
    def save_model(self, epoch, best_epoch, best_model_weights, best_acc, centroids=None):
        
        model_states = {'epoch': epoch,
                'best_epoch': best_epoch,
                'state_dict_best': best_model_weights,
                'best_acc': best_acc,
                'centroids': centroids}

        model_dir = os.path.join(self.training_opt['log_dir'], 
                                 'final_model_checkpoint.pth')

        xm.save(model_states, model_dir)
            
    def output_logits(self, openset=False):
        filename = os.path.join(self.training_opt['log_dir'], 
                                'logits_%s'%('open' if openset else 'close'))
        xm.master_print("Saving total logits to: %s.npz" % filename)
        np.savez(filename, 
                 logits=self.total_logits.detach().cpu().numpy(), 
                 labels=self.total_labels.detach().cpu().numpy(),
                 paths=self.total_paths)
