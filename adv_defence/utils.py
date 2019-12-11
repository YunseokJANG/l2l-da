from __future__ import print_function

import os
import json
import logging
import numpy as np
from datetime import datetime


def get_time():
    return datetime.now().strftime("%m%d_%H%M%S")


def model_name_generator(config):
    name_str = []
    name_str.append('dataset={}'.format(config.dataset))
    name_str.append('lambda={}'.format(config.dsgan_lambda))
    name_str.append('g_method={}'.format(config.g_method))
    name_str.append('mini_step={}'.format(config.g_ministep_size))
    name_str.append('g_z_dim={}'.format(config.g_z_dim))
    name_str.append('cond_batch={}'.format(config.num_classes > 1))
    name_str.append('f_update={}'.format(config.f_update_style))
    name_str.append('g_mini_update={}'.format(config.g_mini_update_style))
    name_str.append('f={}'.format(config.f_classifier_name))
    name_str.append('f_pre={}'.format(config.f_pretrain))
    name_str.append('g_ce={}'.format(config.use_cross_entropy_for_g))
    name_str.append('g_grad={}'.format(config.g_normalize_grad))
    name_str.append('g_step={}'.format(config.train_g_iter))
    name_str.append('f_lr={}'.format(config.f_lr))
    name_str.append('g_lr={}'.format(config.g_lr))
    name_str.append('g_use_grad={}'.format(config.g_use_grad))
    name_str.append('gamma={}'.format(config.lr_gamma))
    name_str.append('b={}'.format(config.single_batch_size))
    name_str.append(get_time())
    return '-'.join(name_str)


def prepare_dirs_and_logger(config):
    formatter = logging.Formatter("%(asctime)s:%(levelname)s::%(message)s")
    logger = logging.getLogger()

    for hdlr in logger.handlers:
        logger.removeHandler(hdlr)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    if config.load_path:
        config.model_dir = config.load_path
        config.model_name = os.path.basename(config.load_path)
    else:
        config.model_name = model_name_generator(config)

    if (not hasattr(config, 'model_dir')) or (len(config.model_dir) == 0):
        config.model_dir = os.path.join(config.log_dir, config.model_name)

    if ('CIFAR10' in config.f_classifier_name) or ('MNIST' in config.f_classifier_name) or ('TinyImagenet' in config.f_classifier_name):
        config.model_dir = os.path.join(config.log_dir, config.f_classifier_name)
        config.model_name = config.f_classifier_name + '_adapt'

    config.data_path = os.path.join(config.data_dir, config.dataset)

    for path in [config.log_dir, config.model_dir]:
        if not os.path.exists(path):
            os.makedirs(path)


def save_config(config):
    param_path = os.path.join(config.model_dir, "params.json")

    print("[*] Model Name: %s" % config.model_name)
    print("[*] PARAM path: %s" % param_path)

    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)
