#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 17:19:50 2019

@author: eric
"""

# EXTERNAL LIBRARY IMPORTS
import prebuilt_loss_functions as plf
import adversarial_training as advtrain
import adversarial_evaluation as adveval
import adversarial_perturbations as ap
import adversarial_attacks as aa
import loss_functions as lf
import argparse
import re
import torch
import os
import torch.nn as nn

from adv_defence.models import Classifier
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
import utils.pytorch_utils as utils


transform_list = [transforms.ToTensor()]
transform_chain = transforms.Compose(transform_list)
image_dataset = datasets.ImageFolder('./datasets/tinyimagenet/val', transform_chain)
test_dataloader = data.DataLoader(image_dataset, batch_size=100, shuffle=False, num_workers=2)

def main(config):
    model = Classifier(200, classifier_name='resnet18', dataset="tinyimagenet", pretrained=False)

    # format matching
    data_classifier_state = torch.load(os.path.join(config.path, 'Classifier.pth'), map_location=None)

    if 'state_dict' in data_classifier_state:
        data_classifier_state = data_classifier_state['state_dict']

    bad_classifier_state = {}
    for k, v in data_classifier_state.items():
        if k.startswith('1.'):
            bad_classifier_state[k[2:]] = v
        else:
            bad_classifier_state[k] = v

    starts_with_module = False
    for key in bad_classifier_state.keys():
        if key.startswith('module.'):
            starts_with_module = True
            break
    if starts_with_module:
        correct_classifier_state = {k[7:]: v for k, v in
                                   bad_classifier_state.items()}
    else:
        correct_classifier_state = bad_classifier_state

    starts_with_feature_extractor = False
    for k in correct_classifier_state.keys():
        if k.startswith('feature_extractor.'):
            starts_with_feature_extractor = True
            break
    if not starts_with_feature_extractor:
        correct_classifier_state = {'feature_extractor.'+k: v for k, v in
                                    correct_classifier_state.items()}

    # fit into our model
    model.load_state_dict( correct_classifier_state )

    normalizer = utils.IdentityNormalize()

    # Put this into the AdversarialEvaluation object
    adv_eval_object = adveval.AdversarialEvaluation(model, normalizer)

    surrogate = model
    normalizer_surr = normalizer

    # First let's build the attack parameters for each.
    # we'll reuse the loss function:
    attack_loss = plf.VanillaXentropy(surrogate, normalizer_surr)
    linf_8_threat = ap.ThreatModel(ap.DeltaAddition, {'lp_style': 'inf',
                                                     'lp_bound': 8.0 / 255.0})

    #------ FGSM Block
    fgsm_attack = aa.FGSM(surrogate, normalizer_surr, linf_8_threat, attack_loss)
    fgsm_attack_kwargs = {'step_size': 8.0 / 255.0,
                           'verbose': False}
    fgsm_attack_params = advtrain.AdversarialAttackParameters(fgsm_attack,
                                                               attack_specific_params=
                                                               {'attack_kwargs': fgsm_attack_kwargs})

    # ------ pgd10 Block
    pgd10_attack = aa.PGD(surrogate, normalizer_surr, linf_8_threat, attack_loss)
    pgd10_attack_kwargs = {'step_size': 8.0/255.0/4.0,
                          'num_iterations': 10,
                          'keep_best': True,
                          'random_init': True,
                          'verbose': False}
    pgd10_attack_params = advtrain.AdversarialAttackParameters(pgd10_attack,
                                                              attack_specific_params=
                                                              {'attack_kwargs': pgd10_attack_kwargs})

    # ------ pgd100 Block
    pgd100_attack = aa.PGD(surrogate, normalizer_surr, linf_8_threat, attack_loss)
    pgd100_attack_kwargs = {'step_size': 8.0/255.0/12.0,
                          'num_iterations': 100,
                          'keep_best': True,
                          'random_init': True,
                          'verbose': False}
    pgd100_attack_params = advtrain.AdversarialAttackParameters(pgd100_attack,
                                                              attack_specific_params=
                                                              {'attack_kwargs': pgd100_attack_kwargs})

    # ------ CarliniWagner100 Block
    cwloss6 = lf.CWLossF6
    distance_fxn = lf.SoftLInfRegularization
    cw100_attack = aa.CarliniWagner(surrogate, normalizer_surr, linf_8_threat, distance_fxn, cwloss6)
    cw100_attack_kwargs = {'num_optim_steps': 100,
                          'verbose': False}
    cw100_attack_params = advtrain.AdversarialAttackParameters(cw100_attack,
                                                              attack_specific_params=
                                                              {'attack_kwargs': cw100_attack_kwargs})

    # ------ CarliniWagner1000 Block
    cwloss6 = lf.CWLossF6
    distance_fxn = lf.SoftLInfRegularization
    cw1000_attack = aa.CarliniWagner(surrogate, normalizer_surr, linf_8_threat, distance_fxn, cwloss6)
    cw1000_attack_kwargs = {'num_optim_steps': 1000,
                          'verbose': False}
    cw1000_attack_params = advtrain.AdversarialAttackParameters(cw1000_attack,
                                                              attack_specific_params=
                                                              {'attack_kwargs': cw1000_attack_kwargs})


    to_eval_dict = {'top1': 'top1',
                    'avg_loss_value': 'avg_loss_value',
                    'avg_successful_ssim': 'avg_successful_ssim'}

    fgsm_eval = adveval.EvaluationResult(fgsm_attack_params,
                                          to_eval=to_eval_dict)

    pgd10_eval = adveval.EvaluationResult(pgd10_attack_params,
                                         to_eval=to_eval_dict)

    pgd100_eval = adveval.EvaluationResult(pgd100_attack_params,
                                         to_eval=to_eval_dict)

    cw100_eval = adveval.EvaluationResult(cw100_attack_params,
                                         to_eval=to_eval_dict)

    cw1000_eval = adveval.EvaluationResult(cw1000_attack_params,
                                         to_eval=to_eval_dict)


    attack_ensemble = {'fgsm': fgsm_eval,
                       'pgd10' : pgd10_eval,
                       'pgd100' : pgd100_eval,
                       'cw100' : cw100_eval,
                       'cw1000' : cw1000_eval}
    ensemble_out = adv_eval_object.evaluate_ensemble(test_dataloader, attack_ensemble,
                                                     verbose=True,
                                                     num_minibatches=None)

    sort_order = {'ground': 1, 'fgsm': 2, 'pgd10': 3, 'pgd100': 4, 'cw100': 5, 'cw1000': 6}
    # sort_order = {'ground': 1, 'pgd10': 2, 'pgd100': 3}
    def pretty_printer(fd, eval_ensemble, result_type):
        print('~' * 10, result_type, '~' * 10)
        fd.write('~' * 10+ result_type+ '~' * 10+"\n")
        for key in sorted(list(eval_ensemble.keys()), key=lambda k: sort_order[k]):
            eval_result = eval_ensemble[key]
            pad = 6 - len(key)
            if result_type not in eval_result.results:
                continue
            avg_result = eval_result.results[result_type].avg
            print(key, pad* ' ', ': ', avg_result)
            fd.write(key + pad* ' '+ ': '+ str(avg_result)+"\n")

    with open(os.path.join(config.path, 'base_eval_result.txt'), "w") as fd:
        fd.write('Result for {}'.format(config.path)+"\n")
        fd.write("\n")
        pretty_printer(fd, ensemble_out, 'top1')
        # We can examine the loss (noting that we seek to 'maximize' loss in the adversarial example domain)
        pretty_printer(fd, ensemble_out, 'avg_loss_value')
        # This is actually 1-SSIM, which can serve as a makeshift 'similarity index',
        # which essentially gives a meterstick for how similar the perturbed images are to the originals
        pretty_printer(fd, ensemble_out, 'avg_successful_ssim')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='tinynet_resnet18_ours_x_x',
                         help='classifier path')
    parser.add_argument('--architecture', type=str, default='resnet18',
                         help='architecture of the model')
    config, _ = parser.parse_known_args()
    main(config)
