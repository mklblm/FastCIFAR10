# Fast CIFAR10 by a beginner

### Background

This repo contains an approximation of David C. Page's implementation for training a ResNet on CIFAR10. In 2018 David wrote a series of blog posts on his investigation into the efficient setup and training of a Residual network, trying to achieve a high classification accuracy. These blog posts are now only accessible through the Internet Archive; links the individual posts are available below.

In November of 2018, Davids implementation was able to achieve a 94% classification accuracy after only 79 seconds of training on a single NVIDIA Volta V100 GPU - a world record at the time. Within a year he managed to reduce training time to 26 seconds. In November of 2023, tysam-code improved the world record to 6.3 seconds on a A100 GPU, which was then improved to 3.29 seconds in March of 2024 by Keller Jordan. The implementations of the latter 2 significantly differ from David's; they feature an improved architecture and some newly found training and code optimisation tricks. 

The implementation featured in this repo is however primarily based on Thomas Germer's work, who is able to achieve 94% validation set accuracy on an A100 in ~14 seconds. Thomas' version has nicely readable code that is technically quite similar to David's original. I also have taken some cues from both tysam-code and Keller Jordan's work. 

### Installation & Usage

This implementation was built with Torch 2.2.2 and Torchvision 0.17.2 and can (probably) only run if your machine has a cuda device. To install dependencies and run a single training run:

```bash
git clone http://github.com/mklblm/fastcifar10
cd fastcifar10
pip install requirements.txt
python train.py
```

As an alternative, you can do a training run on Google Colab: [This] google colab notebook.

### Example result

The results below are cherry-picked  training runs on a Windows desktop with a RTX 3080 GPU and Google Colab with a A100 GPU. As it stands, the current version of my implementation is **not** able to consistently achieve over 94% classification accuracy on the validation set. 

**Windows Desktop (RTX 3080):**

```
PyTorch version: 2.2.2+cu121
Torchvision version: 0.17.2+cu121

Device: cuda
GPU: NVIDIA GeForce RTX 3080

Loading and pre-processing dataset
Files already downloaded and verified
Files already downloaded and verified
Preprocessing time: 3.86 seconds

epoch    batch    train time [sec]    validation accuracy
---------------------------------------------------------
    1       97                5.08                 0.2584
    2      194                8.96                 0.4377
    3      291               12.83                 0.7541
    4      388               16.68                 0.8702
    5      485               20.56                 0.9043
    6      582               24.50                 0.9126
    7      679               28.37                 0.9081
    8      776               32.11                 0.9312
    9      873               35.93                 0.9333
   10      970               39.82                 0.9407
```

**Google Colab (A100):**

```
PyTorch version: 2.4.1+cu121
Torchvision version: 0.19.1+cu121

Device: cuda
GPU: NVIDIA A100-SXM4-40GB

Files already downloaded and verified
Files already downloaded and verified
Preprocessing time: 1.95 seconds

epoch    batch    train time [sec]    validation accuracy
---------------------------------------------------------
    1       97                1.49                 0.2325
    2      194                2.97                 0.3365
    3      291                4.44                 0.7620
    4      388                5.92                 0.8871
    5      485                7.40                 0.9022
    6      582                8.88                 0.9127
    7      679               10.35                 0.9237
    8      776               11.83                 0.9244
    9      873               13.31                 0.9371
   10      970               14.79                 0.9411
```


#### 100 Run Statistics

Although this implementation should be functionally quite close to Thomas Germer's implementation, there are at the very least (possibly significant) differences between the Torch SGD optimizer and his implementation of Gradient Descent using Nesterov velocity momentum. 

Over 100 test runs on a Google Colab A100, i was not able to achieve consistently 94%+ classification accuracy. Only 2 of 100 runs result in accuracy over 94% (compared to Thomas Germer's 84/100 runs achieving this.) If consistent 94%+ average performance (over 100 runs) is even possible on this repo's implementation, further tuning of the optimizer parameters is most certainly necessary. 

100 Run statistics:
**Max accuracy: 0.9411**
**Min accuracy:  0.9224**
**Mean accuracy: 0.9344 (std: 0.0032)**

[insert graphs]

The variance in the early epochs seems to suggest that a more performant learning curve is definitely possible to achieve here. Comparing the standard deviation of the mean to results of other implementations suggest that a higher consistency in convergence is also achievable.

### Motivation

As a Deep Learning and PyTorch novice, my primary goal for this project was to replicate David's and Thomas' work in Python 3.10 / PyTorch 2.2 / Torchvision 0.17. In doing so, i wanted to mostly use functions and classes that are already part of the PyTorch package. This enabled me to gain an understanding of the functionality of David's and Thomas' work, but also to gain some more hands-on experience with PyTorch. 

As a secondary goal, i tried to build this implementation such that i would minimise the loss in both wall time training performance and classification accuracy. However, looking at some of the other implementations mentioned above, it is clear that PyTorch-native code may not be the most efficient solution for certain components. 

As this is my first foray into publishing code on github, I've attempted to post something that is sufficiently and correctly attributed, replicable, and informative. I invite readers to give me feedback by opening a New issue. 


### Considerations

Over the course of this project I quickly learned that numerous aspects of the network architecture, training setup and optimisation tricks are well above my "pay grade" at this point in my career as an aspiring ML engineer. Achieving world-record performance on this task requires one to be well-versed in a number of topics in both general computer science and machine learning. Instead of parroting information I only superficially understand, I implore those interested to read Keller Jordan's paper on his implementation - especially the Methods section. His explanation of the used techniques is well-written. For those seeking more historical context, Internet Archive versions of David's posts are linked below. 


### References

**David C. Page** 
code: https://github.com/davidcpage/cifar10-fast
and blog posts: (Some of these are missing graphics)
* Introduction: https://web.archive.org/web/20181112163539/https://www.myrtle.ai/2018/09/24/how_to_train_your_resnet/
* Baseline Model: https://web.archive.org/web/20181108225027mp_/https://www.myrtle.ai/2018/09/24/how_to_train_your_resnet_1
* Mini Batches: https://web.archive.org/web/20181108225028mp_/https://www.myrtle.ai/2018/09/24/how_to_train_your_resnet_2
* Regularisation: https://web.archive.org/web/20181108225029mp_/https://www.myrtle.ai/2018/09/24/how_to_train_your_resnet_3
* Architecture: https://web.archive.org/web/20181108225030mp_/https://www.myrtle.ai/2018/09/24/how-to-train-your-resnet-4
* Hyperparameters: https://web.archive.org/web/20230131062636/https://myrtle.ai/learn/how-to-train-your-resnet-5-hyperparameters/
* Weight Decay: https://web.archive.org/web/20230401222343/https://myrtle.ai/learn/how-to-train-your-resnet-6-weight-decay/
* Batch Norm: https://web.archive.org/web/20230131071115/https://myrtle.ai/learn/how-to-train-your-resnet-7-batch-norm/
* Bag of Tricks: https://web.archive.org/web/20230210004747/https://myrtle.ai/learn/how-to-train-your-resnet-8-bag-of-tricks/

**Thomas Germer** 
code: https://github.com/99991/cifar10-fast-simple

**Fern** (tysam-code)
code: https://github.com/tysam-code/hlb-CIFAR10

**Keller Jordan** 
code: https://github.com/KellerJordan/cifar10-airbench
paper: https://arxiv.org/abs/2404.00498
