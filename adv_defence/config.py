#-*- coding: utf-8 -*-
import argparse

def str2bool(v):
    return v.lower() in ('true', '1')

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--g_base_channel_dim', type=int, default=64,
                     help='base channel dimension for the Generator network')
net_arg.add_argument('--g_lr', type=float, default=0.01)
net_arg.add_argument('--g_optimizer', type=str, default='sgd')
net_arg.add_argument('--g_momentum', type=float, default=0.9)
net_arg.add_argument('--g_deeper_layer', type=str2bool, default=False)
net_arg.add_argument('--g_beta1', type=float, default=0.5)
net_arg.add_argument('--g_beta2', type=float, default=0.999)
net_arg.add_argument('--g_label_condition_channel', type=int, default=10)
net_arg.add_argument('--g_z_dim', type=int, default=8)


net_arg.add_argument('--f_classifier_name', type=str, default='resnet20', help='classifier network name')
net_arg.add_argument('--f_lr', type=float, default=0.01)
net_arg.add_argument('--f_optimizer', type=str, default='sgd')
net_arg.add_argument('--f_momentum', type=float, default=0.9)
net_arg.add_argument('--f_beta1', type=float, default=0.5)
net_arg.add_argument('--f_beta2', type=float, default=0.999)
net_arg.add_argument('--f_pretrain', type=str2bool, default=False)
net_arg.add_argument('--g_method', type=int, default=3, help='1: L2L, 2: Ours_FGSM, 3: Ours_PGD')
net_arg.add_argument('--f_update_style', type=int, default=1, help='1: update all, 2: update true/false')
net_arg.add_argument('--g_mini_update_style', type=int, default=2, help='0: ce_loss(every) + DS_loss(last), 1: ce_loss(last) + DS_loss(last), 2: ce_loss(every) + DS_loss(every), 3: ce_loss(last) + DS_loss(every)')
net_arg.add_argument('--g_ministep_size', type=float, default=0.25)
net_arg.add_argument('--g_normalize_grad', type=str2bool, default=True)
net_arg.add_argument('--g_use_grad', type=str2bool, default=True)
net_arg.add_argument('--lr_gamma', type=float, default=0.1)
net_arg.add_argument('--sync_batch', type=str2bool, default=True)



# Data
data_arg = add_argument_group('Data')
# MNIST
data_arg.add_argument('--dataset', type=str, default='mnist')
data_arg.add_argument('--single_batch_size', type=int, default=100)
data_arg.add_argument('--num_classes', type=int, default=10)
data_arg.add_argument('--is_rgb', type=str2bool, default=False)
data_arg.add_argument('--img_size', type=int, default=28)
data_arg.add_argument('--epsilon', type=float, default=0.3)


# CIFAR10
#data_arg.add_argument('--dataset', type=str, default='cifar10')
#data_arg.add_argument('--single_batch_size', type=int, default=200)
#data_arg.add_argument('--num_classes', type=int, default=10)
#data_arg.add_argument('--is_rgb', type=str2bool, default=True)
#data_arg.add_argument('--img_size', type=int, default=32)
#data_arg.add_argument('--epsilon', type=float, default=0.03137)


# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--is_train', type=str2bool, default=True)
train_arg.add_argument('--need_samples', type=str2bool, default=False)
train_arg.add_argument('--max_step', type=int, default=100000)
train_arg.add_argument('--weight_decay', type=float, default=0.00001)
train_arg.add_argument('--dsgan_lambda', type=float, default=4.0, help='negative means not using ours')
train_arg.add_argument('--train_f_iter', type=int, default=5)
train_arg.add_argument('--train_g_iter', type=int, default=10)
train_arg.add_argument('--use_cross_entropy_for_g', type=str2bool, default=True)
train_arg.add_argument('--test_save_orig', type=str2bool, default=False)
train_arg.add_argument('--test_save_adv', type=str2bool, default=False)
train_arg.add_argument('--test_iter_steps', type=int, default=10) # no 100?
train_arg.add_argument('--test_cnw_steps', type=int, default=1000)
train_arg.add_argument('--test_cnw_search_steps', type=int, default=6)


# Test specific parameters
test_arg = add_argument_group('Testing')
test_arg.add_argument('--test_arch', type=str, default='20', help='resnet number [18,20,56]')
test_arg.add_argument('--test_classifier_name', type=str, default='Plain', help='[Plain, FSGM, PGD, PGD100, CW, or.. __here__.pth]')
test_arg.add_argument('--test_ckp_num_epoch', type=int, default=100, help='100, 300, or 98999')
test_arg.add_argument('--test_generator_name', type=str, default='NoiseGen_98999', help=' __here__.pth')
test_arg.add_argument('--test_iter', type=int, default=10, help='10 if you want to make it 10 step prediction')
test_arg.add_argument('--test_num_z_samples', type=int, default=10, help='number of samples to generate (stochasticity)')
test_arg.add_argument('--test_dir', type=str, default='loss_eval')
test_arg.add_argument('--l2l_dir', type=str, help='dir_path_for_l2l')


# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--load_path', type=str, default='')
misc_arg.add_argument('--log_step', type=int, default=200)
misc_arg.add_argument('--save_step', type=int, default=3000)
misc_arg.add_argument('--log_level', type=str, default='INFO', choices=['INFO', 'DEBUG', 'WARN'])
misc_arg.add_argument('--log_dir', type=str, default='logs')
misc_arg.add_argument('--data_dir', type=str, default='datasets')
misc_arg.add_argument('--pretrained_dir', type=str, default='./pretrained_models')
misc_arg.add_argument('--num_gpu', type=int, default=1)
misc_arg.add_argument('--random_seed', type=int, default=422)
misc_arg.add_argument('--num_workers', type=int, default=2)


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
