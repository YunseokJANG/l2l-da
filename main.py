import torch

from adv_defence.trainer import Trainer
from adv_defence.classifier_tester import ClassifierTester

from adv_defence.config import get_config
from adv_defence.utils import prepare_dirs_and_logger, save_config
from adv_defence.data_loader import get_loader
import numpy as np
import random

def main(config):
    prepare_dirs_and_logger(config)

    np.random.seed(config.random_seed)
    random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    if config.num_gpu > 0:
        torch.cuda.manual_seed(config.random_seed)

    train_data_loader = get_loader(dataset_name=config.dataset,
                                   root=config.data_path,
                                   batch_size=config.single_batch_size,
                                   split='train',
                                   num_workers=config.num_workers,
                                   shuffle=True)
    test_data_loader = get_loader(dataset_name=config.dataset,
                                  root=config.data_path,
                                  batch_size=config.single_batch_size,
                                  split='test',
                                  num_workers=config.num_workers,
                                  shuffle=False)

    if not config.is_train:
        if config.need_samples:
            toolkit = Trainer(config, None, test_data_loader)
            toolkit.get_sample_pdf_of_checkpoint()
        else:
            evaluator = ClassifierTester(config, test_data_loader)
            evaluator.test_classifier_with_l2lda_att()
    else:
        trainer = Trainer(config, train_data_loader, test_data_loader)
        save_config(config)
        trainer.train()


if __name__ == "__main__":
    config, unparsed = get_config()
    main(config)
