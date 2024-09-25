### Utilities for Fast CIFAR10 Training

import torch
from torchvision import datasets
from torchvision.transforms import v2


## Patch-based PCA Whitening weights initialisation 
# implementation by Thomas Germer: https://github.com/99991/cifar10-fast-simple
# Proposed by David Page: https://web.archive.org/web/20201123223831/https://myrtle.ai/learn/how-to-train-your-resnet-8-bag-of-tricks/

def patch_whitening(data, patch_size=(3, 3)):
    """
    Compute weights from data such that torch.std(F.conv2d(data, weights), dim=(2, 3))is close to 1.
    """

    h, w = patch_size
    c = data.size(1)
    patches = data.unfold(2, h, 1).unfold(3, w, 1)
    patches = patches.transpose(1, 3).reshape(-1, c, h, w).to(torch.float32)

    n, c, h, w = patches.shape
    X = patches.reshape(n, c * h * w)
    X = X / (X.size(0) - 1) ** 0.5
    covariance = X.t() @ X

    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.flip(0)
    eigenvectors = eigenvectors.t().reshape(c * h * w, c, h, w).flip(0)

    return eigenvectors / torch.sqrt(eigenvalues + 1e-2).view(-1, 1, 1, 1)



## Data loading and pre-processing:

def preprocess_data(dataset, dtype=torch.float16, device="cuda"):
    """
    Change dataset datatype to dtype, then send to GPU. 
    Calculate mean and std from dataset, then normalize. 
    Permute data dimensions from NHWC to NCWH. 
    Pad Training Images.
    """

    # Cast dataset data to tensor of dtype, place on GPU
    data = torch.from_numpy(dataset.data).to(dtype).to(device)

    # Normalize data
    # type and to might be redundant here
    mean = torch.mean(data, axis=(0,1,2))
    std = torch.std(data, axis=(0,1,2))
    data = (data - mean) / std

    # permute data NHWC to NCWH
    # See: https://discuss.pytorch.org/t/why-does-pytorch-prefer-using-nchw/83637/4
    data = data.permute(0, 3, 1, 2)

    # pad training images
    # functional.pad adds padding to 2 last dims of tensor, so perform after permute
    if dataset.train:
        data = v2.functional.pad(data, 4, padding_mode="reflect")

    return data

def load_data(data_path="data", dtype=torch.float16, device="cuda"):
    """
    Load CIFAR10 dataset. 
    Preprocess train and Validation sets.
    """
    
    print("Loading and pre-processing dataset")

    train = datasets.CIFAR10(root=data_path, train=True, download=True)
    test = datasets.CIFAR10(root=data_path, train=False, download=True)

    X_train = preprocess_data(train, dtype, device)
    X_test = preprocess_data(test, dtype, device)

    y_train = torch.tensor(train.targets).to(device)
    y_test = torch.tensor(test.targets).to(device)

    return X_train, y_train, X_test, y_test


## Learning Rate schedule helper

def lr_lambda_helper(linear_factor: float, phase1: float, phase2:float, total_train_steps: int):
    """
    Helper function for LambdaLR scheduler. Schedule LR in 3 separate phases.
    
    Scales defined LR in 3 phases:
    Phase 1: Increasing LR for (p1*total_steps) steps, by factor 0 to 1.
    Phase 2: Decreasing LR for (p2*total_steps) steps, by factor 1 to linear_factor.
    Phase 3: Constant LR at scale linear_factor.

    Returns Lambda LR scheduling helper. 
    """
    
    p1 = int(phase1*total_train_steps)
    p2 = int(phase2*total_train_steps)

    def helper(step):
        if step < p1:                       # Phase 1
            return step / p1    
        elif step < (p1 + p2):              # Phase 2
            return 1 - ((step - p1) * (1 - linear_factor) / p2)
        else:
            return linear_factor            # Phase 3: Constant LR
        
    return helper