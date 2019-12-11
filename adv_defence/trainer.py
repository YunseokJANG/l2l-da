""" trainer.py """
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
from glob import glob
from tqdm import trange
from itertools import chain
from tensorboardX import SummaryWriter

from adv_defence.models import *
from adv_defence.attacks import get_fgsm, get_pgd, get_cw, run_fgsm, run_pgd, run_cw
from adv_defence.sync_batchnorm import DataParallelWithCallback

class Trainer(object):
    def __init__(self, config, train_data_loader, test_data_loader):
        self.config = config
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.start_step = 0
        self.tensorboard = None
        self._build_model()

        if config.num_gpu > 0:
            self.NoiseGenerator = DataParallelWithCallback(self.NoiseGenerator.cuda(),
                                                  device_ids=range(config.num_gpu))
            self.Classifier = DataParallelWithCallback(self.Classifier.cuda(),
                                              device_ids=range(config.num_gpu))

        # # Note: check whether :0 is nessasary or not.
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if config.load_path:
            self._load_model()

        # create the attacker modules
        self.FGSM = get_fgsm(self.config.dataset)
        self.PGD = get_pgd(self.config.dataset)
        self.CW = get_cw(self.config.dataset)


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
        self.NoiseGenerator.apply(weights_init_normal)
        if not self.config.f_pretrain:
            self.Classifier.apply(weights_init_normal)


    def _load_model(self):
        print("[*] Load models from {}...".format(self.config.load_path))
        paths = glob(os.path.join(self.config.load_path, 'Classifier_*.pth'))
        paths.sort()

        if len(paths) == 0:
            path = os.path.join(self.config.load_path, 'Classifier.pth')
            if not os.path.exists(path):
                print("[!] No checkpoint found in {}...".format(self.config.load_path))
                return
            self.start_step = 0
        else:
            idxes = [int(os.path.basename(path.split('.')[-2].split('_')[-1])) for path in paths]
            self.start_step = max(idxes)

        if self.config.num_gpu == 0:
            map_location = lambda storage, loc: storage
        else:
            map_location = None


        if self.config.f_update_style != -1:
            bad_classifier_state = torch.load('{}/Classifier_{}.pth'.format(self.config.load_path, self.start_step),
                                              map_location=map_location)
            starts_with_module = False
            for key in bad_classifier_state.keys():
                if key.startswith('module.'):
                    starts_with_module = True
                    break
            if starts_with_module and (self.config.num_gpu < 1):
                correct_classifier_state = {k[7:]: v for k, v in
                                           bad_classifier_state.items()}
            else:
                correct_classifier_state = bad_classifier_state
            self.Classifier.load_state_dict( correct_classifier_state )

        if self.config.f_update_style != -1:
            bad_generator_state = torch.load('{}/NoiseGen_{}.pth'.format(self.config.load_path, self.start_step),
                                             map_location=map_location)
        else:
            bad_generator_state = torch.load('{}/Generator.pth'.format(self.config.load_path),
                                             map_location=map_location)

        starts_with_module = False
        for key in bad_generator_state.keys():
            if key.startswith('module.'):
                starts_with_module = True
                break
        if starts_with_module and (self.config.num_gpu < 1):
            correct_generator_state = {k[7:]: v for k, v in
                                       bad_generator_state.items()}
        else:
            correct_generator_state = bad_generator_state
        self.NoiseGenerator.load_state_dict( correct_generator_state )


    def _save_model(self, step):
        print("[*] Save models to {}...".format(self.config.model_dir))
        torch.save(self.Classifier.state_dict(),
                   '{}/Classifier_{}.pth'.format(self.config.model_dir, step))
        torch.save(self.NoiseGenerator.state_dict(),
                   '{}/NoiseGen_{}.pth'.format(self.config.model_dir, step))


    def _merge_noise(self, sum_noise, cur_noise, eps_step, eps_all):
        # 0. normalize noise output first: Don't need to, since we always take the tanh output

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
        return torch.mean(weight * output)


    def _compute_acc(self, logits, labels):
        #logits = logits / torch.norm(logits)
        _max_val, max_idx = torch.max(logits, 1)
        return torch.mean(torch.eq(max_idx, labels).double())


    # compute our loss from the output (batch major!)
    def _dsgan_loss(self, noise, output, single_batch, stability=1e-8):
        if noise is None:
            return None

        numerator = torch.mean(torch.abs(output[:single_batch] - output[single_batch:]),
                               dim=[_ for _ in range(1, len(output.shape))])
        denominator = torch.mean(torch.abs(noise[:single_batch] - noise[single_batch:]),
                                 dim=[_ for _ in range(1, len(noise.shape))])
        our_term = torch.mean(numerator / (denominator + stability))
        return our_term


    def train(self):
        # Optimizer for G
        if self.config.g_optimizer == 'adam':
            g_optimizer = torch.optim.Adam(self.NoiseGenerator.parameters(),
                                           lr=self.config.g_lr,
                                           betas=(self.config.g_beta1, self.config.g_beta2),
                                           weight_decay=self.config.weight_decay)
        elif self.config.g_optimizer == 'sgd':
            g_optimizer = torch.optim.SGD(self.NoiseGenerator.parameters(),
                                          lr=self.config.g_lr,
                                          momentum=self.config.g_momentum,
                                          weight_decay=self.config.weight_decay)
        else:
            raise Exception("[!] Optimizer for the generator should be ['adam', 'sgd']")

        # set initial learning rate for the case that it starts training from the middle
        if self.config.f_update_style == 2:
            if self.start_step != 0:
                for group in g_optimizer.param_groups:
                    group.setdefault('initial_lr', self.config.g_lr)
            g_scheduler = torch.optim.lr_scheduler.StepLR(g_optimizer,
                                                          step_size = self.config.max_step // 2,
                                                          gamma = self.config.lr_gamma,
                                                          last_epoch = (-1 if self.start_step == 0 else self.start_step))
        else:
            g_scheduler = None



        # Optimizer for F
        if self.config.f_optimizer == 'adam':
            f_optimzer = torch.optim.Adam(self.Classifier.parameters(), lr=self.config.f_lr,
                                          betas=(self.config.f_beta1, self.config.f_beta2),
                                          weight_decay=self.config.weight_decay)
        elif self.config.f_optimizer == 'sgd':
            f_optimizer = torch.optim.SGD(self.Classifier.parameters(), lr=self.config.f_lr,
                                          momentum=self.config.f_momentum,
                                          weight_decay=self.config.weight_decay)
        else:
            raise Exception("[!] Optimizer for the generator should be ['adam', 'sgd']")

        f_scheduler = torch.optim.lr_scheduler.StepLR(f_optimizer,
                                                      step_size = self.config.max_step // 2,
                                                      gamma = self.config.lr_gamma,
                                                      last_epoch = (-1 if self.start_step == 0 else self.start_step))
        if self.start_step != 0:
            for group in f_optimizer.param_groups:
                group.setdefault('initial_lr', self.config.f_lr)




        # now load the train data
        loader = iter(self.train_data_loader)

        # train mode
        self.tensorboard = SummaryWriter(self.config.model_dir)
        self.tensorboard.add_text(tag='argument', text_string=str(self.config.__dict__))
        for step in trange(self.start_step, self.config.max_step, ncols=80):
            try:
                data = loader.next()
            except StopIteration:
                loader = iter(self.train_data_loader)
                data = loader.next()

            # convert unit to float
            real_img = self._get_variable(data[0].type(torch.FloatTensor))
            if (not self.config.is_rgb) and (len(real_img.shape) == 3):
                real_img = torch.unsqueeze(real_img, 1) # N W H -> N C W H
            label = self._get_variable(data[1].type(torch.LongTensor))
            single_batch_size = label.size(0)


            # try to reduce the learning rate of f
            if f_scheduler is not None:
                f_scheduler.step()
            if g_scheduler is not None:
                g_scheduler.step()


            # MNIST w/ lenet case only:
            # (pretrain 1K steps to make the classifier to be trained)
            # For all the other cases, we've loaded the pretrained hyperparameters
            # If you have a pretrained weights, then you can start from there.
            if (step < 1000) and (self.config.f_classifier_name == 'lenet'):
                self.Classifier.train()
                self.Classifier.zero_grad()
                class_output = self.Classifier(real_img)
                cls_loss = self._cross_entropy_loss(class_output, label, single_batch_size)
                cls_loss.backward()
                f_optimizer.step()
                continue


            ######## Phase 1 #######
            # Grab gradients from f before training the G
            # obtain the gradient from the classifier
            self.Classifier.eval()
            self.Classifier.zero_grad()
            grad_input = real_img.detach()
            grad_input.requires_grad = True
            class_output = self.Classifier.forward(grad_input)


            # compute loss
            # f_loss is always being averaged when it computed, so don't need to be re-scaled.
            cls_loss = self._cross_entropy_loss(class_output, label,
                                              single_batch_size)
            # Add other losses here if you want.
            grad_loss = cls_loss


            if self.config.g_use_grad:
                # obtain gradients and disable the gradient for the input
                grad_loss.backward()
                f_grad = grad_input.grad
                # normalized the gradient input
                # Please change it to other normalization if needed
                if self.config.g_normalize_grad:
                    f_grad_norm = f_grad + 1e-15 # DO NOT EDIT! Need a stabilizer in here!!!
                    f_grad = f_grad / f_grad_norm.norm(dim=(2,3), keepdim=True)

                f_grad = f_grad.detach() # for a sanity check purpose



            ######## Phase 2 ###########
            # Train the generator, not the discriminator.
            # But, discriminator is still required, in order to compute the gradient
            double_real_img = torch.cat((real_img, real_img), 0).detach()
            double_label = torch.cat((label, label), 0).detach()

            if self.config.g_method % 2 == 1:
                double_adv_sum = torch.zeros_like(double_real_img)
            else:
                double_adv_sum = None


            if self.config.g_use_grad:
                double_adv_grad = torch.cat((f_grad, f_grad), 0)
            else:
                double_adv_grad = None


            if self.config.g_z_dim > 0:
                if self.config.num_gpu > 0:
                    g_z = torch.cuda.FloatTensor(single_batch_size * 2, self.config.g_z_dim).normal_()
                else:
                    g_z = torch.FloatTensor(single_batch_size * 2, self.config.g_z_dim).normal_()
            else:
                g_z = None

            self.NoiseGenerator.train()
            self.Classifier.eval()
            self.NoiseGenerator.zero_grad()
            proxy_loss_sum = 0.0
            dsgan_loss_sum = 0.0
            update_list = []

            if self.config.g_mini_update_style not in [0,1,2,3]:
                raise Exception("[!] g_mini_update_style should be in [0,1,2,3]")


            for g_iter_step_no in range(self.config.train_g_iter):
                if not self.config.use_cross_entropy_for_g:
                    print("[!] We cannot train our generator without cross_entropy")
                    break

                # generate the current pair
                img_grad_noise = double_real_img

                if self.config.g_use_grad:
                    img_grad_noise = torch.cat((double_real_img, double_adv_grad), 1)

                # in case of recursive gen
                if self.config.g_method == 3:
                    img_grad_noise = torch.cat((img_grad_noise, double_adv_sum), 1)


                # feed it to the generator
                noise_output_for_g = self.NoiseGenerator(img_grad_noise, double_label, g_z)


                # clamping learned noise in epsilon boundary
                if self.config.g_method % 2 == 1:
                    clamp_noise = self._merge_noise(double_adv_sum, noise_output_for_g,
                            self.config.epsilon * self.config.g_ministep_size,
                            self.config.epsilon)
                else:
                    clamp_noise = self.config.epsilon * noise_output_for_g

                # clamping it once again to image boundary
                adv_img_for_g = torch.clamp(double_real_img.detach() + clamp_noise,
                                  0.0, 1.0)


                # first obtain the gradient information for the current result
                copy_for_grad = adv_img_for_g.detach()
                copy_for_grad.requires_grad = True


                # compute & accumulate classification gradients
                if (self.config.g_mini_update_style % 2 == 0) or (g_iter_step_no + 1 == self.config.train_g_iter):
                    self.Classifier.zero_grad()
                    noise_class_output_for_g = self.Classifier.forward(adv_img_for_g)
                    # compute gradient for the generator
                    # - cross entropy will increase the confidence quickly
                    # - cw loss can find the attack region effectively (fast).
                    proxy_loss = 0.0
                    if self.config.use_cross_entropy_for_g:
                        ce_loss = self._cross_entropy_loss(noise_class_output_for_g,
                                                           double_label,
                                                           single_batch_size)
                        proxy_loss -= ce_loss

                    # add other losses if you want.
                    proxy_loss_sum += proxy_loss


                # compute & accumulate DSGAN gradient
                if (self.config.g_z_dim > 0) and \
                        ((self.config.g_mini_update_style >= 2) or (g_iter_step_no + 1 == self.config.train_g_iter)):

                    # compute our loss and add it to g_loss
                    dsgan_magnitude = self._dsgan_loss(g_z, adv_img_for_g, single_batch_size)
                    if self.config.dsgan_lambda > 0.0:
                        dsgan_loss = -1.0 * self.config.dsgan_lambda * dsgan_magnitude
                    else:
                        dsgan_loss = 0.0
                    dsgan_loss_sum += dsgan_loss


                # preparing for the next mini-step
                if g_iter_step_no + 1 != self.config.train_g_iter:
                    if self.config.g_use_grad:
                        # compute gradient information for the next time step
                        self.Classifier.zero_grad()
                        grad_output_for_g = self.Classifier.forward(copy_for_grad)
                        grad_ce_loss = self._cross_entropy_loss(grad_output_for_g,
                                                                double_label,
                                                                single_batch_size)
                        grad_loss = grad_ce_loss
                        grad_loss.backward()


                        # obtain gradients and disable the gradient for the input
                        f_grad = copy_for_grad.grad
                        # normalized the gradient input.
                        # Please change it to other normalization if needed
                        if self.config.g_normalize_grad:
                            f_grad_norm = f_grad + 1e-15 # DO NOT EDIT! Need a stabilizer in here!!!
                            f_grad = f_grad / f_grad_norm.norm(dim=(2,3), keepdim=True)
                        double_adv_grad = f_grad.detach()

                    if double_adv_sum is not None:
                        double_adv_sum = clamp_noise

                update_list.append(adv_img_for_g.detach())


            g_loss_sum = proxy_loss_sum + dsgan_loss_sum
            g_loss_sum.backward()
            # https://github.com/pytorch/examples/blob/master/word_language_model/main.py
            nn.utils.clip_grad_norm_(self.NoiseGenerator.parameters(), 1.0)
            g_optimizer.step()



            ######## Phase 3 #########
            # train the Discriminator
            if self.config.f_update_style == 1:
                # merge update
                f_label_list = [torch.cat((label, label, label), 0)]
                f_update_list = [torch.cat((real_img, update_list[-1]), 0)]

            elif self.config.f_update_style == 2:
                # update false labels first
                # then update the true label
                f_label_list = [double_label, label]
                f_update_list = [update_list[-1], real_img]
            elif self.config.f_update_style == -1:
                # finetune our generator only
                if (step % self.config.save_step) == (self.config.save_step - 1):
                    self._save_model(step)
                    self.defence_regular_eval(iter_step=step)
                continue
            else:
                raise Exception("[!] f_update_style should be [1: single, 2: twice]")

            self.Classifier.train()
            noise_class_output_for_debugging = None
            noise_class_loss_for_debugging = None
            real_pred_sum = 0.0
            fake_pred_sum = 0.0
            for image_for_f, label_for_f in zip(f_update_list, f_label_list):
                self.Classifier.zero_grad()

                noise_class_output = self.Classifier(image_for_f)
                if noise_class_output_for_debugging is None:
                    noise_class_output_for_debugging = noise_class_output

                cls_loss = self._cross_entropy_loss(noise_class_output,
                                                    label_for_f,
                                                    single_batch_size)
                if noise_class_loss_for_debugging is None:
                    noise_class_loss_for_debugging = cls_loss

                f_loss = cls_loss
                # update the classifier and the generator
                f_loss.backward()
                # https://github.com/pytorch/examples/blob/master/word_language_model/main.py
                nn.utils.clip_grad_norm_(self.Classifier.parameters(), 1.0)
                f_optimizer.step()


            ######## Logging ##########
            f_acc = self._compute_acc(class_output[-single_batch_size:], label).data
            self.tensorboard.add_scalar('train/f_loss', f_loss.data, step)
            self.tensorboard.add_scalar('train/f_acc', f_acc, step)
            if self.config.dsgan_lambda > 0.0:
                self.tensorboard.add_scalar('train/lambda', self.config.dsgan_lambda, step)

            fg_acc = self._compute_acc(noise_class_output_for_debugging[-single_batch_size*2:], double_label).data
            fg_loss = self._cross_entropy_loss(noise_class_output_for_debugging[-single_batch_size*2:],
                                               double_label, single_batch_size)
            self.tensorboard.add_scalar('train/fg_cls_loss', fg_loss.data, step)
            self.tensorboard.add_scalar('train/fg_acc', fg_acc, step)
            if step % self.config.log_step == 0:
                print("")
                print("[{}/{}] Acc_F: {:.4f} F_loss: {:.4f} Acc_FG: {:.4f} cls_loss: {:.4f}".\
                       format(step, self.config.max_step, f_acc, f_loss.data, fg_acc, fg_loss.data))


            if self.config.train_g_iter > 0:
                self.tensorboard.add_scalar('train/proxy_loss_sum', proxy_loss_sum.data, step)
                self.tensorboard.add_scalar('train/proxy_loss_last', proxy_loss.data, step)
                if (self.config.g_z_dim > 0) and (dsgan_magnitude is not None):
                    if self.config.dsgan_lambda > 0.0:
                        self.tensorboard.add_scalar('train/dsgan_loss_sum', dsgan_loss_sum.data, step)
                    self.tensorboard.add_scalar('train/dsgan_loss_last', dsgan_magnitude.data, step)
                    if step % self.config.log_step == 0:
                        print("[{}/{}] our_loss: {:.4f} proxy_loss: {:.4f}". \
                              format(step, self.config.max_step, dsgan_magnitude.data, proxy_loss.data))
                self.tensorboard.add_scalar('train/g_loss_sum', g_loss_sum.data, step)


            # save checkpoints and noise image
            if step % self.config.save_step == self.config.save_step - 1:

                if self.config.g_use_grad:
                    slice1 = update_list[-1][:single_batch_size]
                    slice2 = update_list[-1][single_batch_size:]

                    grad_abs = torch.abs(double_adv_grad)
                    grad_min = torch.min(    grad_abs   )
                    grad_rescale = grad_abs - grad_min
                    grad_max = torch.max(grad_rescale)
                    grad_rescale /= grad_max
                    grad_slice1 = grad_rescale[:single_batch_size]
                    grad_slice2 = grad_rescale[single_batch_size:]
                    self.tensorboard.add_image('train/pair1',
                                               tvutils.make_grid(torch.cat((real_img[:15], slice1[:15], slice2[:15], grad_slice1[:15], grad_slice2[:15]), 0), nrow = 15),
                                               step)
                    self.tensorboard.add_image('train/pair2',
                                               tvutils.make_grid(torch.cat((real_img[15:30], slice1[15:30], slice2[15:30], grad_slice1[15:30], grad_slice2[15:30]), 0), nrow = 15),
                                               step)

                self._save_model(step)
                self.defence_regular_eval(iter_step=step)





    def _test_classifier(self, image_tensor, label_tensor, iter_step=0, method_name='PGD'):
        total_acc_f = []
        num_items = len(label_tensor)
        self.Classifier.eval()
        for index in range(0, num_items, self.config.single_batch_size):
            # first slice into batch
            adv_img = image_tensor[index:min(index+self.config.single_batch_size,num_items)]
            label = label_tensor[index:min(index+self.config.single_batch_size,num_items)]

            # run classifier
            logits = self.Classifier.forward(adv_img)

            # get accuracy
            acc_f = self._compute_acc(logits, label)
            total_acc_f.append(acc_f.data)

        # aggregate the performance
        performance = sum(total_acc_f) / len(total_acc_f)
        print("[{} / {}] Acc: {:.4f}".format(method_name, iter_step, performance))


        if self.tensorboard is not None:
            self.tensorboard.add_scalar('test/{}_acc'.format(method_name), performance, iter_step)




    def get_sample_pdf_of_checkpoint(self, default_z_iter=10):
        loader = iter(self.test_data_loader)

        test_dir = os.path.join(self.config.model_dir, 'test')
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        self.Classifier.eval()
        self.NoiseGenerator.eval()
        total_acc_f = []
        total_acc_g = []
        real_img_arr = []
        real_label_arr = []
        adv_img_arr = []
        adv_att_arr = []

        for step in trange(len(self.test_data_loader), ncols=80):
            try:
                data = loader.next()
            except StopIteration:
                print("[!] Test sample generation finished. Samples are in {}".format(test_dir))
                break

            real_img = self._get_variable(data[0].type(torch.FloatTensor))
            if (not self.config.is_rgb) and (len(real_img.shape) == 3):
                real_img = torch.unsqueeze(real_img, 1)

            label = self._get_variable(data[1].type(torch.LongTensor))
            single_batch_size = label.size(0)


            ######## Phase 1 #######
            # Grab gradient from f before training the G
            self.Classifier.zero_grad()
            grad_input = real_img.detach()
            grad_input.requires_grad = True
            class_output = self.Classifier.forward(grad_input)


            # compute loss
            f_loss = self._cross_entropy_loss(class_output, label, single_batch_size)
            f_loss.backward()


            if self.config.g_use_grad:
                f_grad = grad_input.grad
                if self.config.g_normalize_grad:
                    f_grad_norm = f_grad + 1e-15 # DO NOT EDIT! Need a stabilizer in here!!!
                    f_grad = f_grad / f_grad_norm.norm(dim=(2,3), keepdim=True)


            # Phase 2 #
            num_iter_z = default_z_iter if self.config.g_z_dim > 0 else 1
            adv_img_inner_arr = []
            adv_att_inner_arr = []
            for _ in range(num_iter_z):
                adv_grad = f_grad.detach()

                if self.config.g_method % 2 == 1:
                    adv_sum = torch.zeros_like(real_img)
                else:
                    adv_sum = None


                if self.config.g_z_dim > 0:
                    if self.config.num_gpu > 0:
                        g_z = torch.cuda.FloatTensor(single_batch_size, self.config.g_z_dim).normal_()
                    else:
                        g_z = torch.FloatTensor(single_batch_size, self.config.g_z_dim).normal_()
                else:
                    g_z = None


                self.NoiseGenerator.zero_grad()
                for g_iter_step_no in range(self.config.train_g_iter):
                    img_grad_noise = real_img
                    if self.config.g_use_grad:
                        img_grad_noise = torch.cat((img_grad_noise, adv_grad), 1)

                    if self.config.g_method == 3:
                        img_grad_noise = torch.cat((img_grad_noise, adv_sum), 1)

                    # feed it to the generator
                    noise_output = self.NoiseGenerator.forward(img_grad_noise, label, g_z)

                    # generate learned noise
                    if self.config.g_method % 2 == 1:
                        clamp_noise = self._merge_noise(adv_sum, noise_output,
                                                        self.config.epsilon * self.config.g_ministep_size,
                                                        self.config.epsilon)
                    else:
                        clamp_noise = self.config.epsilon * noise_output
                    adv_img_for_g = torch.clamp(real_img.detach() + clamp_noise,
                                                0.0, 1.0)
                    copy_for_grad = adv_img_for_g.detach()
                    copy_for_grad.requires_grad = True

                    # preparing for the next mini-step
                    # Note: we are not updating the Generator.
                    if g_iter_step_no + 1 != self.config.train_g_iter:
                        if self.config.g_use_grad:
                            self.Classifier.zero_grad()
                            grad_output_for_g = self.Classifier.forward(copy_for_grad)
                            grad_ce_loss = self._cross_entropy_loss(grad_output_for_g,
                                                                    label,
                                                                    single_batch_size)
                            grad_loss = grad_ce_loss
                            grad_loss.backward()

                            # obtain gradients and disable the gradient for the input
                            f_inner_grad = copy_for_grad.grad
                            if self.config.g_normalize_grad:
                                f_inner_grad_norm = f_inner_grad + 1e-15 # DO NOT EDIT! Need a stabilizer in here!!!
                                f_inner_grad = f_inner_grad / f_inner_grad_norm.norm(dim=(2,3), keepdim=True)
                            adv_grad = f_inner_grad.detach()

                        if adv_sum is not None:
                            adv_sum = clamp_noise

                    # generate learned noise
                    target_image = adv_img_for_g.detach()
                    target_attack = clamp_noise.detach()

                adv_img_inner_arr.append(target_image.detach().data)
                adv_att_inner_arr.append(target_attack.detach().data)

            self.Classifier.zero_grad()
            class_output = self.Classifier.forward(real_img)
            noise_class_output = self.Classifier.forward(target_image)
            acc_f = self._compute_acc(class_output, label)
            acc_g = self._compute_acc(noise_class_output, label)
            total_acc_f.append(acc_f.data)
            total_acc_g.append(acc_g.data)

            real_img_arr.append(real_img.unsqueeze(1).detach().data)
            real_label_arr.append(label.data)
            adv_img_arr.append(torch.transpose(torch.stack(adv_img_inner_arr), 0, 1))
            adv_att_arr.append(torch.transpose(torch.stack(adv_att_inner_arr), 0, 1))

        print("[{}] Acc_F: {:.4f}, Acc_FG: {}".format(test_dir,
                                                      sum(total_acc_f) / len(total_acc_f),
                                                      sum(total_acc_g) / len(total_acc_g)))

        print("Converting the results into numpy format.")
        real_img_arr = torch.cat(real_img_arr, 0)
        orig_data_cpu = real_img_arr.mul(255).clamp(0, 255).byte().permute(0, 1, 3, 4, 2).cpu().numpy()

        real_label_arr = torch.cat(real_label_arr, 0)
        orig_label_cpu = real_label_arr.to(dtype=torch.int16).cpu().numpy()

        adv_img_arr = torch.cat(adv_img_arr, 0)
        adv_img_cpu = adv_img_arr.mul(255).clamp(0, 255).byte().permute(0, 1, 3, 4, 2).cpu().numpy()

        adv_att_arr = torch.clamp((1.0 + torch.cat(adv_att_arr, 0) / self.config.epsilon) / 2.0, 0.0, 1.0)
        adv_att_cpu = adv_att_arr.mul(255).clamp(0, 255).byte().permute(0, 1, 3, 4, 2).cpu().numpy()

        print("start generating a pdf file")
        item_dict_for_pdf = {}
        for real, label, img, att in zip(orig_data_cpu, orig_label_cpu, adv_img_cpu, adv_att_cpu):
            current_std = np.reshape(att[4:], [6, -1])
            current_std = np.expand_dims(current_std, 1) - np.expand_dims(current_std, 0)
            current_std = np.mean(np.sum(current_std * current_std, axis=-1) * (1 - np.eye(6)))

            temp_arr = np.concatenate([img[4:], att[4:]], axis=0)
            temp_arr = np.transpose(temp_arr, (1, 0, 2, 3))
            shape = temp_arr.shape

            if (label not in item_dict_for_pdf) or (item_dict_for_pdf[label][0] < current_std):
                if shape[3] == 1:
                    item_dict_for_pdf[label] = [current_std, np.reshape(temp_arr, (shape[0], shape[1] * shape[2]))]
                else:
                    item_dict_for_pdf[label] = [current_std, np.reshape(temp_arr, (shape[0], shape[1] * shape[2], shape[3]))]

        sorted_list = [item_dict_for_pdf[_][1] for _ in range(self.config.num_classes)]
        output = np.concatenate(sorted_list, axis=0)


        print("start saving it in {} as vis_{}.pdf".format(self.config.log_dir, self.config.model_name))
        import scipy.misc
        scipy.misc.imsave(os.path.join(self.config.log_dir, 'vis_{}.pdf'.format(self.config.model_name)), output)


    def _run_single_attack(self, iter_step=0, method_name='PGD'):
        # set test dir to save
        test_dir = os.path.join(self.config.model_dir, 'test')
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        # set a new data_loader
        loader = iter(self.test_data_loader)
        steps_required_per_epoch = len(loader)
        if method_name.endswith('_slow'):
            steps_required_per_epoch = 5

        print(steps_required_per_epoch)
        print('[Info] Start running {} for step {}'.format(method_name, iter_step))

        # run attack mechanism
        output_list = []
        target_list = []
        self.Classifier.eval()
        for step in range(steps_required_per_epoch):
            try:
                data = loader.next()
            except StopIteration:
                loader = iter(self.test_data_loader)
                data = loader.next()

            # convert unit to float
            input_img = self._get_variable(data[0].type(torch.FloatTensor))
            if (not self.config.is_rgb) and (len(input_img.shape) == 3):
                input_img = torch.unsqueeze(input_img, 1)
            target_label = self._get_variable(data[1].type(torch.LongTensor))
            single_batch_size = target_label.size(0)

            if method_name == 'FGSM':
                adv_result = run_fgsm(self.FGSM, self.Classifier, input_img, target_label,
                                      self.config.epsilon)
            elif method_name == 'PGD':
                adv_result = run_pgd(self.PGD, self.Classifier, input_img, target_label,
                                     self.config.epsilon, self.config.test_iter_steps)
            elif method_name == 'CW':
                adv_result = run_cw(self.CW, self.Classifier, input_img, target_label)
            elif method_name == 'ORIGINAL':
                adv_result = input_img

            output_list.append( adv_result )
            target_list.append( target_label )

        output_tensor = torch.cat(output_list, dim=0)
        label_tensor = torch.cat(target_list, dim=0)

        if self.config.test_save_adv:
            np.save('{}/attack_{}_step{}_img.npy'.format(test_dir, method_name, iter_step),
                    output_tensor.permute(0, 2, 3, 1).cpu().numpy())
            np.save('{}/attack_{}_step{}_label.npy'.format(test_dir, method_name, iter_step),
                    label_tensor.permute(0, 2, 3, 1).cpu().numpy())

        self._test_classifier(output_tensor, label_tensor, iter_step, method_name)


    def defence_regular_eval(self, iter_step=0):
        # set classifier to be in evaluation mode
        self.Classifier.eval()

        self._run_single_attack(iter_step, 'FGSM')
        self._run_single_attack(iter_step, 'PGD')
        # self._run_single_attack(iter_step, 'CW')
        self._run_single_attack(iter_step, 'ORIGINAL')

        # return back to train mode
        self.Classifier.train()
        return


    def defence_over_cnw(self, iter_step=0):
        # assume model is loaded properly
        self.Classifier.eval()

        self._run_single_attack(iter_step, 'CW')

        self.Classifier.train()
        return


    def _get_variable(self, inputs):
        if self.config.num_gpu > 0:
            out = Variable(inputs.cuda())
        else:
            out = Variable(inputs)
        return out
