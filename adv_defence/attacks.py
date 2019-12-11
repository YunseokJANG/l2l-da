import loss_functions as lf
import prebuilt_loss_functions as plf
import utils.pytorch_utils as dn
import adversarial_attacks as aa
import adversarial_perturbations as ap


def _get_settings(dataset='mnist'):
    if dataset == 'mnist':
        normalizer = None
        eps = 0.3
    elif dataset == 'cifar10':
        normalizer = None
        eps = 8.0 / 255.0
    elif dataset == 'tinyimagenet':
        normalizer = None
        eps = 8.0 / 255.0
    else:
        raise Exception("[!] dataset should be ['mnist', 'cifar10', 'tinyimagenet']")
    return eps, normalizer


##########################################################################
#   FGSM ATTACK                                                          #
##########################################################################
def get_fgsm(dataset='mnist'):
    eps, normalizer = _get_settings(dataset)
    delta_threat = ap.ThreatModel(ap.DeltaAddition,
                                  ap.PerturbationParameters(lp_style='inf',
                                                            lp_bound=eps,
                                                            manual_gpu=True))
    return aa.FGSM(classifier_net=None, normalizer=normalizer,
                   threat_model=delta_threat, loss_fxn=None, manual_gpu=True)


def run_fgsm(fgsm_obj, model, img, targets, eps):
    fgsm_obj.classifier_net = model
    fgsm_obj.loss_fxn = plf.VanillaXentropy(model, fgsm_obj.normalizer)
    fgsm_output = fgsm_obj.attack(img, targets, eps, verbose=False)
    return fgsm_output.adversarial_tensors()


##########################################################################
#   PGD ATTACK                                                           #
##########################################################################
def get_pgd(dataset='mnist'):
    eps, normalizer = _get_settings(dataset)
    delta_threat = ap.ThreatModel(ap.DeltaAddition,
                                  ap.PerturbationParameters(lp_style='inf',
                                                            lp_bound=eps,
                                                            manual_gpu=True))
    return aa.PGD(classifier_net=None, normalizer=normalizer,
                  threat_model=delta_threat, loss_fxn=None, manual_gpu=True)


def run_pgd(pgd_obj, model, img, targets, eps, iter=10):
    if iter == 10:
        step_size = eps / 4
    else:
        raise Exception("please set an appropriate step size for this case")

    pgd_obj.classifier_net = model
    pgd_obj.loss_fxn = plf.VanillaXentropy(model, pgd_obj.normalizer)
    pgd_output = pgd_obj.attack(img, targets, step_size,
                                num_iterations=iter,
                                random_init=True, signed=True,
                                verbose=False, keep_best=True)
    return pgd_output.adversarial_tensors()


##########################################################################
#   CW Linfty ATTACK                                                     #
##########################################################################
def get_cw(dataset='mnist'):
    eps, normalizer = _get_settings(dataset)
    delta_threat = ap.ThreatModel(ap.DeltaAddition,
                                  ap.PerturbationParameters(lp_style='inf',
                                                            lp_bound=eps,
                                                            manual_gpu=True))
    return aa.CarliniWagner(classifier_net=None,
                            normalizer=normalizer, threat_model=delta_threat,
                            distance_fxn=lf.SoftLInfRegularization,
                            carlini_loss=lf.CWLossF6, manual_gpu=True)


def run_cw(cw_obj, model, img, targets, iter=1000, search=5):
    cw_obj.classifier_net = model
    cw_output = cw_obj.attack(img, targets,
                              num_bin_search_steps=search,
                              num_optim_steps=iter,
                              confidence=0.0, initial_lambda=1.0, verbose=False)

    return cw_output.adversarial_tensors()
