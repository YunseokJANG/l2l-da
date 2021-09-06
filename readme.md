# Adversarial Defense via Learning to Generate Diverse Attacks #


This repository includes the codes for our ICCV 2019 paper.
* Yunseok Jang, Tianchen Zhao, Seunghoon Hong and Honglak Lee. [_Adversarial Defense via Learning to Generate Diverse Attacks_](http://openaccess.thecvf.com/content_ICCV_2019/papers/Jang_Adversarial_Defense_via_Learning_to_Generate_Diverse_Attacks_ICCV_2019_paper.pdf). In ICCV, 2019


If you use any of the material in this repository as part of your work, we ask you to cite:

```
@inproceedings{jang-ICCV-2019,
    author    = {Yunseok Jang and Tianchen Zhao and Seunghoon Hong and Honglak Lee},
    title     = {{Adversarial Defense via Learning to Generate Diverse Attacks}},
    booktitle = {ICCV},
    year      = 2019
}
```

Please contact [Yunseok Jang](mailto:yunseokj@umich.edu) if you have any question.



## How to Use ##

### 1. Environment Setup ###
```
conda create -n l2lda -y
source activate l2lda
conda config --add channels conda-forge
conda install -c conda-forge python=3.6.4 numba -y
# conda install -c anaconda cudatoolkit=10.0 cudnn=7.6 -y
conda install pytorch=1.1.0 torchvision=0.3.0 cudatoolkit=10.0 -c pytorch -y
pip install requests gpustat tensorboardX tensorflow-gpu visdom tqdm matplotlib scikit-image

# I've forked and included this code base
# git clone https://github.com/revbucket/mister_ed.git
# Also, this repo includes edited version of Synced Batch Norm from this repository
# git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
```

Then, add [mister_ed](https://github.com/revbucket/mister_ed.git) package setting to conda environment.
```
conda install conda-build
cd ./mister_ed
conda develop .

# Download pretrained ResNet Classifier trained on CIFAR-10 dataset
python scripts/setup_cifar.py
mv pretrained_models/ ../
cd ..
```

Optional) Download our pretrained classifiers.
```
# Note: For CIFAR 10 model, please check the detailed architecture in mister_ed/cifar10/cifar_resnets.py
wget https://www.dropbox.com/s/cnw79x2mv4qffw4/evaluation.zip -O evaluation.zip --no-check-certificate
unzip evaluation.zip
```

Optional) If you want to play with Tiny Imagenet dataset, you need to preprocess the dataset via the following commands.
```
mkdir datasets
cd datasets
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip
rm -f tiny-imagenet-200.zip
mv tiny-imagenet-200 tinyimagenet
cd ../
python tiny_imagenet_val_format.py
```



### 2. Train a new classifier ###
a. MNIST
```
CUDA_VISIBLE_DEVICES=0 python main.py --f_classifier_name='lenet' --g_optimizer=sgd --g_lr=0.01 --f_pretrain=False --g_method=3 --train_g_iter=10 --g_ministep_size=0.25 --dsgan_lambda=0.02 --g_mini_update_style=2 --f_update_style=1 --num_gpu=1
```

b. CIFAR-10
```
CUDA_VISIBLE_DEVICES=0 python main.py --f_classifier_name='resnet20' --dataset=cifar10 --is_rgb=True --img_size=32 --epsilon=0.031372549 --g_optimizer=sgd --g_lr=0.01 --f_pretrain=True --g_method=3 --train_g_iter=10 --g_ministep_size=0.25 --dsgan_lambda=0.5 --g_mini_update_style=2 --f_update_style=1 --num_gpu=1
```

c. Tiny Imagenet
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --f_classifier_name='resnet18' --dataset=tinyimagenet --is_rgb=True --img_size=64 --num_classes=200 --single_batch_size=60 --epsilon=0.031372549 --g_optimizer=sgd --g_lr=0.025 --f_lr=0.005 --f_pretrain=True --g_method=3 --train_g_iter=10 --g_ministep_size=0.25 --dsgan_lambda=0.3 --g_mini_update_style=2 --f_update_style=1 --num_gpu=4 --lr_gamma=0.3
```
Please note that we did early-stopping based on its test performance. Also, please carefully check the configurations if you want to change any.



### 3. Measure the classification performance of a classifier ###
a. MNIST
```
CUDA_VISIBLE_DEVICES=0 python evaluation_mnist.py --path=./evaluation/mnist_lenet_ours
```

b. CIFAR-10
```
CUDA_VISIBLE_DEVICES=0 python evaluation_cifar.py --path=./evaluation/cifar_resnet20_ours
```

c. Tiny Imagenet
```
CUDA_VISIBLE_DEVICES=0 python evaluation_tinyimagenet.py --path=./evaluation/tinynet_resnet18_ours
```




### 4. White-box attack via L2L-DA generator ###
a. MNIST
```
# first finetune our generator to adapt
CUDA_VISIBLE_DEVICES=0 python main.py --f_classifier_name=OurLeNetMNIST --g_optimizer=sgd --g_lr=0.01 --f_pretrain=True --g_method=3 --train_g_iter=10 --g_ministep_size=0.25 --dsgan_lambda=0.02 --g_mini_update_style=2 --f_update_style=-1 --num_gpu=1 --lr_gamma=1.0 --max_step=48000 --log_dir=logs/finetune --load_path=evaluation/mnist_lenet_ours


# then try to attack the classifier
CUDA_VISIBLE_DEVICES=0 python main.py --f_classifier_name=OurLeNetMNIST --test_generator_name=logs/finetune/OurLeNetMNIST/NoiseGen_47999.pth --dataset=mnist --test_dir=attack_eval --is_train=False --need_samples=False --is_rgb=False --img_size=28 --epsilon=0.3 --g_optimizer=sgd --g_lr=0.01 --f_update_style=-1 --f_pretrain=True --train_g_iter=10 --test_iter=10 --g_ministep_size=0.25 --dsgan_lambda=0.02 --g_mini_update_style=2 --g_method=3 --num_classes=10 --g_z_dim=8 --num_gpu=1 --lr_gamma=1.0
```

b. CIFAR-10
```
# first finetune our generator to adapt
CUDA_VISIBLE_DEVICES=0 python main.py --f_classifier_name=OurResNet20CIFAR10 --dataset=cifar10 --is_rgb=True --img_size=32 --epsilon=0.031372549 --g_optimizer=sgd --g_lr=0.01 --f_pretrain=True --g_method=3 --train_g_iter=10 --g_ministep_size=0.25 --dsgan_lambda=0.5 --g_mini_update_style=2 --f_update_style=-1 --num_gpu=1 --lr_gamma=1.0 --max_step=48000 --log_dir=logs/finetune --load_path=evaluation/cifar_resnet20_ours

# then try to attack the classifier
CUDA_VISIBLE_DEVICES=0 python main.py --f_classifier_name=OurResNet20CIFAR10 --test_generator_name=logs/finetune/OurResNet20CIFAR10/NoiseGen_47999.pth --dataset=cifar10 --test_dir=attack_eval --is_train=False --need_samples=False --is_rgb=True --img_size=32 --epsilon=0.031372549 --g_optimizer=sgd --g_lr=0.01 --f_update_style=-1 --f_pretrain=True --train_g_iter=10 --test_iter=10 --g_ministep_size=0.25 --dsgan_lambda=0.5 --g_mini_update_style=2 --g_method=3 --num_classes=10 --g_z_dim=8 --num_gpu=1 --lr_gamma=1.0
```

c. Tiny Imagenet
```
# first finetune our generator to adapt
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --f_classifier_name=OurResNet18TinyImagenet --dataset=tinyimagenet --is_rgb=True --img_size=64 --num_classes=200 --single_batch_size=60 --epsilon=0.031372549 --g_optimizer=sgd --g_lr=0.025 --f_lr=0.005 --f_pretrain=True --g_method=3 --train_g_iter=10 --g_ministep_size=0.25 --dsgan_lambda=0.3 --g_mini_update_style=2 --f_update_style=-1 --num_gpu=4 --lr_gamma=1.0 --max_step=48000 --log_dir=logs/finetune --load_path=evaluation/tinynet_resnet18_ours


# then try to attack the classifier
CUDA_VISIBLE_DEVICES=0 python main.py --f_classifier_name=OurResNet18TinyImagenet --test_generator_name=logs/finetune/OurResNet18TinyImagenet/NoiseGen_47999.pth --dataset=tinyimagenet --test_dir=attack_eval --is_train=False --need_samples=False --is_rgb=True --img_size=64 --num_classes=200 --epsilon=0.031372549 --g_optimizer=sgd --g_lr=0.025 --f_lr=0.005 --f_update_style=-1 --f_pretrain=True --train_g_iter=10 --test_iter=10 --g_ministep_size=0.25 --dsgan_lambda=0.3 --g_mini_update_style=2 --g_method=3 --g_z_dim=8 --num_gpu=1 --lr_gamma=1.0
```







### 5. Visualize adversarial noises ###
a. MNIST
```
CUDA_VISIBLE_DEVICES=0 python main.py --is_train=False --need_samples=True --f_classifier_name=OurLeNetMNIST --is_rgb=False --img_size=28 --g_optimizer=sgd --g_lr=0.01 --f_update_style=-1 --f_pretrain=True --train_g_iter=10 --g_ministep_size=0.25 --dsgan_lambda=0.02 --g_mini_update_style=2 --g_method=3 --num_classes=10 --g_z_dim=8 --num_gpu=1 --load_path=evaluation/mnist_lenet_ours
```

b. CIFAR-10
```
CUDA_VISIBLE_DEVICES=0 python main.py --is_train=False --need_samples=True --f_classifier_name=OurResNet20CIFAR10 --dataset=cifar10 --is_rgb=True --img_size=32 --epsilon=0.031372549 --g_optimizer=sgd --g_lr=0.01 --f_update_style=-1 --f_pretrain=True --train_g_iter=10 --g_ministep_size=0.25 --dsgan_lambda=0.5 --g_mini_update_style=2 --g_method=3 --num_classes=10 --g_z_dim=8 --num_gpu=1 --load_path=evaluation/cifar_resnet20_ours
```
