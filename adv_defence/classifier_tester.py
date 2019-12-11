""" classifier_tester.py """
from __future__ import print_function
from __future__ import division

import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.parallel
import torchvision.utils as tvutils
from torch.autograd import Variable

import os
import numpy as np
from tqdm import trange
from tensorboardX import SummaryWriter

from adv_defence.models import NoiseGenerator, Classifier, weights_init_normal
import mister_ed.utils.checkpoints as checkpoints
import re
from datetime import datetime
from adv_defence.sync_batchnorm import DataParallelWithCallback
import time


class ClassifierTester(object):
    def __init__(self, config, test_data_loader):
        self.config = config
        self.test_data_loader = test_data_loader
        self.start_step = 0
        self.tensorboard = None
        self._build_model()

        if config.num_gpu > 0:
            if config.sync_batch:
                self.NoiseGenerator = DataParallelWithCallback(self.NoiseGenerator.cuda(),
                                                      device_ids=range(config.num_gpu))
            else:
                self.NoiseGenerator = nn.DataParallel(self.NoiseGenerator.cuda(),
                                                      device_ids=range(config.num_gpu))
            self.Classifier = nn.DataParallel(self.Classifier.cuda(),
                                              device_ids=range(config.num_gpu))

        self._load_model()


    def _build_model(self):
        noise_channel_size = (3 if self.config.is_rgb else 1) * (1 + (1 if self.config.g_method == 3 else 0) + (1 if self.config.g_use_grad else 0))

        self.NoiseGenerator = NoiseGenerator(self.config.g_base_channel_dim,
                                             noise_channel_size,
                                             self.config.g_z_dim,
                                             self.config.g_deeper_layer,
                                             self.config.num_classes,
                                             3 if self.config.is_rgb else 1)
        self.Classifier = Classifier(num_classes=self.config.num_classes,
                                     classifier_name=self.config.f_classifier_name,
                                     dataset=self.config.dataset,
                                     pretrained=self.config.f_pretrain,
                                     pretrained_dir=self.config.pretrained_dir)


    def _load_model(self):
        if self.config.num_gpu == 0:
            map_location = lambda storage, loc: storage
        else:
            map_location = None
        print('loading')

        if self.config.test_generator_name:
            bad_state_dict = torch.load(self.config.test_generator_name)

            starts_with_module = False
            for key in bad_state_dict.keys():
                if key.startswith('module.'):
                    starts_with_module = True
                    break

            if starts_with_module and (self.config.num_gpu < 1):
                correct_state_dict = {k[7:]: v for k, v in
                                      bad_state_dict.items()}
            else:
                correct_state_dict = bad_state_dict

            self.NoiseGenerator.load_state_dict( correct_state_dict )
            self.NoiseGenerator.cuda()


    def _merge_noise(self, sum_noise, cur_noise, eps_step, eps_all):
        # 0. normalize noise output first: Don't need to, since we take the tanh output
        # 1. multiply epsilon (with randomness for the training)
        # result: noise is in -eps_step < noise < eps_step
        cur_noise = cur_noise * eps_step

        # 2. return mixed output
        return torch.clamp(sum_noise + cur_noise, -1.0 * eps_all, 1.0 * eps_all)


    def _cross_entropy_loss(self, noise_class_output, label, pure_batch, adv_mult=1.0):
        log_prob = F.log_softmax(noise_class_output, dim=1)
        weight = torch.ones_like(label).float()
        weight[pure_batch:] *= adv_mult
        output = F.nll_loss(log_prob, label, reduction='none')
        return torch.mean(weight * output), output


    def _compute_acc(self, logits, labels):
        _max_val, max_idx = torch.max(logits, 1)
        return torch.mean(torch.eq(max_idx, labels).double())


    def _compute_acc_custom(self, logits, labels):
        _max_val, max_idx = torch.max(logits, 1)
        return torch.eq(max_idx, labels).double()


    def test_classifier_with_l2lda_att(self):
        loader = iter(self.test_data_loader)

        test_dir = self.config.test_dir
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        self.Classifier.eval()
        self.NoiseGenerator.eval()
        total_acc_f = []
        total_acc_ours = []
        total_loss_ours = []
        for step in trange(len(self.test_data_loader), ncols=80):
            try:
                data = loader.next()
            except StopIteration:
                print("[!] Test sample generation finished. Samples are in {}".format(test_dir))
                break

            real_img = self._get_variable(data[0].type(torch.FloatTensor))
            if (not self.config.is_rgb) and (len(real_img.shape) == 3):
                real_img = torch.unsqueeze(real_img, 1) # N W H -> N C W H
            label = self._get_variable(data[1].type(torch.LongTensor))
            single_batch_size = label.size(0)

            # random normal
            g_z_list = [torch.cuda.FloatTensor(single_batch_size, self.config.g_z_dim).normal_()
                        for _ in range(self.config.test_num_z_samples)]

            ours_acc_seed_list = []
            ours_loss_seed_list = []
            self.Classifier.zero_grad()
            grad_input = real_img.detach()
            grad_input.requires_grad = True

            class_output_real = self.Classifier.forward(grad_input)
            cls_loss, _ = self._cross_entropy_loss(class_output_real, label,
                                                   single_batch_size, 1.0)
            cls_loss.backward()

            init_f_grad = grad_input.grad

            acc_f = self._compute_acc(class_output_real, label)
            total_acc_f.append(acc_f.data)

            if self.config.g_normalize_grad:
                init_f_grad_norm = init_f_grad + 1e-15 # add a numerical stabilizer
                init_f_grad = init_f_grad / init_f_grad_norm.norm(dim=(2,3), keepdim=True)
            init_f_grad = init_f_grad.detach()

            for g_z in g_z_list:
                adv_sum = torch.zeros_like(real_img)
                adv_grad = init_f_grad.detach()
                loss_collect_ours = []
                for _ in range(int(self.config.test_iter)):
                    self.NoiseGenerator.zero_grad()

                    img_grad_noise = torch.cat((real_img.detach(), adv_grad, adv_sum), 1)
                    noise_output_for_g = self.NoiseGenerator.forward(img_grad_noise, label, g_z)
                    clamp_noise = self._merge_noise(adv_sum, noise_output_for_g,
                                    self.config.epsilon * self.config.g_ministep_size,
                                    self.config.epsilon)
                    adv_img_for_g = torch.clamp(real_img.detach() + clamp_noise,
                                      0.0, 1.0)

                    copy_for_grad = adv_img_for_g.detach()
                    copy_for_grad.requires_grad = True


                    self.Classifier.zero_grad()
                    grad_output_for_g = self.Classifier.forward(copy_for_grad)
                    grad_ce_loss, loss_value = self._cross_entropy_loss(grad_output_for_g,
                                                                        label,
                                                                        single_batch_size, 1.0)
                    grad_loss = grad_ce_loss
                    grad_loss.backward()

                    f_grad = copy_for_grad.grad
                    # normalized the gradient input.
                    if self.config.g_normalize_grad:
                        f_grad_norm = f_grad + 1e-15 # DO NOT EDIT! Need a stabilizer in here!!!
                        f_grad = f_grad / f_grad_norm.norm(dim=(2,3), keepdim=True)
                    adv_grad = f_grad.detach()

                    adv_sum = clamp_noise.detach()
                    loss_collect_ours.append(loss_value.data)
                acc_g = self._compute_acc_custom(grad_output_for_g, label)
                ours_acc_seed_list.append(acc_g.data)

                loss_collect_ours = torch.stack(loss_collect_ours)
                ours_loss_seed_list.append(loss_collect_ours)

            ours_acc_seed_list = torch.stack(ours_acc_seed_list)
            ours_loss_seed_list = torch.stack(ours_loss_seed_list)

            total_acc_ours.append(ours_acc_seed_list)
            total_loss_ours.append(ours_loss_seed_list)

        ########################
        # iter, numz, batch
        total_acc_ours = torch.stack(total_acc_ours)
        total_acc_ours = torch.transpose(total_acc_ours, 0, 1)
        # numz, iter, batch -> num_z, data
        total_acc_ours = total_acc_ours.contiguous().view(self.config.test_num_z_samples, -1)
        acc_best = torch.mean(torch.min(total_acc_ours, dim=0)[0]).data
        acc_worst = torch.mean(torch.max(total_acc_ours, dim=0)[0]).data
        acc_avg = torch.mean(total_acc_ours).data

        total_loss_ours = torch.stack(total_loss_ours)
        total_loss_ours = torch.transpose(total_loss_ours, 0, 2)

        total_loss_ours = total_loss_ours.contiguous().view(self.config.test_iter, self.config.test_num_z_samples, -1)

        ours_max = torch.mean(torch.max(total_loss_ours, dim=1)[0], dim=1).data.cpu().numpy()
        ours_mean = torch.mean(torch.mean(total_loss_ours, dim=1), dim=1).data.cpu().numpy()
        ours_err = torch.mean(torch.std(total_loss_ours, dim=1), dim=1).data.cpu().numpy()

        string = "[{}] Acc_F: {:.4f}, BestAcc_ours: {:.4f}, AvgAcc_Ours: {:.4f}, WorstAcc_Ours: {:.4f}, AvgLoss_Ours: {:.4f}, MaxLoss_Ours: {:.4f}".format(test_dir,
                                      torch.mean(torch.stack(total_acc_f)).data,
                                      acc_best, acc_avg, acc_worst,
                                      ours_mean[-1],
                                      ours_max[-1])

        with open(os.path.join(test_dir, '{}-{}-{}-result.txt'.format(self.config.dataset,
                             self.config.f_classifier_name,
                             self.config.test_iter)), 'w') as f:
            f.write(string + '\n')
            f.write('OURS_MAX'+','.join([str(_) for _ in ours_max]) + '\n')
            f.write('OURS_MEAN'+','.join([str(_) for _ in ours_mean]) + '\n')
            f.write('OURS_ERR'+','.join([str(_) for _ in ours_err]) + '\n')
        print(string)



    def _get_variable(self, inputs):
        if self.config.num_gpu > 0:
            out = Variable(inputs.cuda())
        else:
            out = Variable(inputs)
        return out
