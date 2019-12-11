#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# EXTERNAL LIBRARY IMPORTS
import prebuilt_loss_functions as plf
import cifar10.cifar_loader as cifar_loader
import adversarial_training as advtrain
import adversarial_evaluation as adveval
import adversarial_perturbations as ap 
import adversarial_attacks as aa
import utils.checkpoints as checkpoints
import loss_functions as lf
import argparse
import re
import torch




def main(config):
    defence_method = config.defence
    flavor = config.architecture
    blackbox = config.blackbox
    flavor_blackbox = config.flavor_blackbox
    epoch = config.epoch
#    assert defence_method in ['PLAIN','FGSM', 'PGD', 'CW'],"INVALID ATTACK: %s" % defence_method
    assert flavor in ['20', '56', 'wide'],"INVALID ARCHITECTURE: %s" % flavor
    
    # Load the trained model and normalizer
    if flavor in ['20', '56']:
        model, normalizer = cifar_loader.load_pretrained_cifar_resnet(
                flavor=int(flavor), return_normalizer=True)
    elif flavor == 'wide':
        model, normalizer = cifar_loader.load_pretrained_cifar_wide_resnet(
                return_normalizer=True)
        
    if defence_method in ['FGSM', 'PGD', 'CW', 'PGD40', 'PGD100']:
        model = checkpoints.load_state_dict(
                defence_method+'ResNet'+flavor, 'resnet'+flavor, epoch, model)
    elif defence_method != 'PLAIN':
        bad_state_dict = torch.load('./pretrained_models/'+defence_method+'.pth')
        correct_state_dict = {re.sub(r'^.*feature_extractor\.', '', k): v for k, v in
                              bad_state_dict.items()}
        model.load_state_dict(correct_state_dict)
    
    
    # Load the evaluation dataset 
    cifar_valset = cifar_loader.load_cifar_data('val', no_transform=True, shuffle=False, batch_size=100) 
    
    # Put this into the AdversarialEvaluation object
    adv_eval_object = adveval.AdversarialEvaluation(model, normalizer)
    
    # Use blackbox attack or not
    if blackbox:
        surrogate, normalizer_surr= cifar_loader.load_pretrained_cifar_resnet(
                flavor=int(flavor_blackbox), return_normalizer=True)
        surrogate.cuda()
    else:
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
                          'verbose': False}
    pgd10_attack_params = advtrain.AdversarialAttackParameters(pgd10_attack, 
                                                              attack_specific_params=
                                                              {'attack_kwargs': pgd10_attack_kwargs})
    

    # ------ pgd100 Block 
    pgd100_attack = aa.PGD(surrogate, normalizer_surr, linf_8_threat, attack_loss)
    pgd100_attack_kwargs = {'step_size': 8.0/255.0/12.0, 
                          'num_iterations': 100, 
                          'keep_best': True,
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
    
    '''
    Next we'll build the EvaluationResult objects that wrap these. 
    And let's say we'll evaluate the:
    - top1 accuracy 
    - average loss 
    - average SSIM distance of successful perturbations [don't worry too much about this]
    
    The 'to_eval' dict as passed in the constructor has structure 
     {key : <shorthand fxn>}
    where key is just a human-readable handle for what's being evaluated
    and shorthand_fxn is either a string for prebuilt evaluators, or you can pass in a general function to evaluate
    '''
    
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
                       'cw1000' : cw1000_eval
                      }
    if blackbox:
        attack_ensemble = {'fgsm': fgsm_eval,
                           'pgd10' : pgd10_eval,
                           'pgd100' : pgd100_eval
                          }
    ensemble_out = adv_eval_object.evaluate_ensemble(cifar_valset, attack_ensemble, 
                                                     verbose=True, 
                                                     num_minibatches=None)
    
    filename = "result.txt"
    if blackbox:
        filename = "result_blackbox.txt"
    # Now let's build a little helper to print things out cleanly:
    
    sort_order = {'ground': 1, 'fgsm': 2, 'pgd10': 3, 'pgd100': 4, 'cw100': 5, 'cw1000': 6}
    if blackbox:
        sort_order = {'ground': 1, 'fgsm': 2, 'pgd10': 3, 'pgd100': 4}
    def pretty_printer(eval_ensemble, result_type):
        f = open(filename, "a")
        print('~' * 10, result_type, '~' * 10)
        f.write('~' * 10+ result_type+ '~' * 10+"\n")
        for key in sorted(list(eval_ensemble.keys()), key=lambda k: sort_order[k]):
            eval_result = eval_ensemble[key]
            pad = 6 - len(key)
            if result_type not in eval_result.results:
                continue 
            avg_result = eval_result.results[result_type].avg
            print(key, pad* ' ', ': ', avg_result)
            f.write(key+ pad* ' '+ ': '+ str(avg_result)+"\n")
        f.close()
        
    '''And then we can print out and look at the results:
    This prints the accuracy. 
    Ground is the unperturbed accuracy. 
    If everything is done right, we should see that PGD with an l_inf bound of 4 is a stronger attack 
    against undefended networks than FGSM with an l_inf bound of 8
    '''
    f = open(filename, "a")
    f.write('Result for ' + defence_method + 'ResNet{}'.format(flavor)+"\n")
    if blackbox:
        f.write('Blackbox' + flavor_blackbox+"\n")
    f.close()
    pretty_printer(ensemble_out, 'top1')
    # We can examine the loss (noting that we seek to 'maximize' loss in the adversarial example domain)
    pretty_printer(ensemble_out, 'avg_loss_value')
    # This is actually 1-SSIM, which can serve as a makeshift 'similarity index', 
    # which essentially gives a meterstick for how similar the perturbed images are to the originals
    pretty_printer(ensemble_out, 'avg_successful_ssim')
    f = open(filename, "a")
    f.write("\n")
    f.close()
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--defence', type=str, default='Yunseok/cifar56_l2l',
                         help='adversarially trained with attack methods')
    parser.add_argument('--architecture', type=str, default='56',
                         help='architecture of the model')
    parser.add_argument('--blackbox', type=bool, default=False,
                         help='whether use blackbox or not')
    parser.add_argument('--flavor_blackbox', type=str, default='32',
                         help='architecture of the surrogate')
    parser.add_argument('--epoch', type=int, default=100,
                         help='training epoch of the target model')
    config, _ = parser.parse_known_args()
    main(config)
