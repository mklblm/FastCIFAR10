import torch
from torch import nn

from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

import torchvision

from timeit import default_timer as timer


from utils import *
from network import *


## Train network on CIFAR10
# Training loop structure by Thomas Germer: https://github.com/99991/cifar10-fast-simple/

def train(seed=0):
    ### Model and Training Hyperparameters
    # Follows Page, Germer

    # Model parameters
    output_scale = 1/8
    celu_alpha = 0.3

    # Training length parameters
    epochs = 10
    batch_size = 512

    # Loss function parameters:
    label_smoothing = 0.2
    loss_reduction = "none"

    # Optimizer parameters
    bias_scaler = 64
    lr_weights = 0.512 / batch_size                   # apply linear scaling rule
    lr_bias = 0.512 * bias_scaler/batch_size
    wd_weights = 0.0005 * batch_size
    wd_biases = 0.0005 * bias_scaler/batch_size
    momentum = 0.9
    nesterov = True

    # LambdaLR Scheduler helper parameters for Page-like learning curve
    linear_factor = 0.1
    phase1 = 0.2 
    phase2 = 0.7
    total_train_steps = epochs * int(50000/batch_size)

    # Validation Model EMA update weight every 5 batches
    ema_update_freq = 5
    ema_alpha = 0.99**ema_update_freq

    # set manual seed for reproducibility
    torch.manual_seed(seed)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define Precision for training speedup
    dtype = torch.float16 if device.type != "cpu" else torch.float32

    # Show package versions and hardware on first run
    if seed == 0:
        print(f"PyTorch version: {torch.__version__}\nTorchvision version: {torchvision.__version__}\n")
        print(f"Device: {device}")
        if device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(device.index)}\n")


    # Start Pre-training timer
    pre_train_time_start = timer()

    ## Load dataset
    X_train, y_train, X_test, y_test = load_data()

    ## Model setup
    # initialise weights of first layer with patch-whitening.
    first_layer_weights = patch_whitening(X_train[:, :, 4:-4, 4:-4])

    # Initialise training model
    model = FastCIFAR_BoT(first_layer_weights, celu_alpha=celu_alpha, input_shape=3, hidden_units=64, output_shape=10, output_scale=output_scale)    
    
    # Set model weights to half precision (torch.float16) for faster training
    model.to(dtype)
    
    # Set BatchNorm layers back to single precision (better accuracy)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.float()
    
    # Upload model to GPU
    model.to(device)
    
    # Initialise validation model (Receives EMA weight updates)
    val_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(ema_alpha))

    ## Initialise loss function
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing,reduction=loss_reduction)

    ## Initialise Optimizer and LR Scheduler
    # For separate updating of weights and biases:
    weights = [par for name, par in model.named_parameters() if "weight" in name and par.requires_grad]
    biases = [par for name, par in model.named_parameters() if "bias" in name and par.requires_grad]

    optim_weight = SGD(params=weights, lr=lr_weights, weight_decay=wd_weights, momentum=momentum, nesterov=nesterov)
    optim_bias = SGD(params=biases, lr=lr_bias, weight_decay=wd_biases, momentum=momentum, nesterov=nesterov)

    lr_lambda = lr_lambda_helper(linear_factor,phase1,phase2,total_train_steps)

    lr_sched_w = LambdaLR(optim_weight, lr_lambda=lr_lambda)
    lr_sched_b = LambdaLR(optim_bias, lr_lambda=lr_lambda)


    ## End of pre-processing / pre-training initialisation
    pre_train_time = timer()-pre_train_time_start
    print(f"Preprocessing time: {pre_train_time:.2f} seconds\n")


    ## Training Loop inits
    batch_count = 0
    total_train_time = 0

    # for tracking training time and validation accuracy curves
    epoch_time = []
    epoch_accuracy = []

    # Print stats:
    print("epoch    batch    train time [sec]    validation accuracy")
    print("---------------------------------------------------------")

    for epoch in range(1, epochs+1):
        # Wait for all cuda streams on device to complete
        torch.cuda.synchronize()

        # Start epoch timer
        time_start_epoch = timer()

        ## Regularization transforms:
        # Alternating Flip (Jordan Keller); Can possibly speed up by avoiding torch.cat
        if epoch % 2:
            X_data = v2.functional.horizontal_flip(X_train[:25000])
            X_data = torch.cat((X_data,X_train[25000:]))
        else:
            X_data = v2.functional.horizontal_flip(X_train[25000:])
            X_data = torch.cat((X_train[:25000],X_data))
        
        # Random crop training images to 32x32
        X_data = v2.RandomCrop(size=(32,32))(X_data)

        ## Randomly Erase 8x8 square from training image. Generally lowers validation accuracy.
        #X_data = v2.RandomErasing(p=0.5, scale=(0.25, 0.25), ratio=(1,1))(X_data)

        # Shuffle training data
        idx = torch.randperm(len(X_data), device=device)
        X_data = X_data[idx]
        y_data = y_train[idx]


        # Iterate over batches:
        for i in range(0, len(X_data), batch_size):
            # Discard partial batch (last batch is 336 images)
            if i+batch_size > len(X_data):
                break

            # select batch slice from training data
            X_batch = X_data[i:i+batch_size]
            y_true = y_data[i:i+batch_size]
    
            # Zero gradients before update, set model to train
            model.zero_grad()
            model.train(True)

            # Forward Pass
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_true)

            # Backward pass
            # Summed loss, refer Page
            loss.sum().backward()

            # Optimizer and LR step
            optim_weight.step()
            optim_bias.step()

            lr_sched_w.step()
            lr_sched_b.step()

            # update Validation model parameters with EMA every 5 batches
            if (i // batch_size % ema_update_freq) == 0:
                val_model.update_parameters(model)

            batch_count += 1

        # Add epoch time to total
        total_train_time += timer()-time_start_epoch

        # Validation loop code taken directly from Thomas Germer
        val_correct = []
        for i in range(0, len(X_test), batch_size):
            val_model.train(False)

            # Test time augmentation; Follows Page, Germer
            regular_inputs = X_test[i : i + batch_size]
            flipped_inputs = torch.flip(regular_inputs, [-1])

            # val model logits
            logits1 = val_model(regular_inputs).detach()
            logits2 = val_model(flipped_inputs).detach()

            # Final logits are average of augmented logits
            logits = torch.mean(torch.stack([logits1, logits2], dim=0), dim=0)

            # Compute correct predictions
            correct = logits.max(dim=1)[1] == y_test[i : i + batch_size]

            val_correct.append(correct.detach().type(torch.float64))


        # Report validation accuracy
        val_acc = torch.mean(torch.cat(val_correct)).item()
        print(f"{epoch:5} {batch_count:8d} {total_train_time:19.2f} {val_acc:22.4f}")

        epoch_accuracy.append(val_acc)
        epoch_time.append(total_train_time)

    return val_acc


if __name__ == "__main__":
    train()