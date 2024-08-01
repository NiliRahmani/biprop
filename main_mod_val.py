import os
import pathlib
import random
import time
import pickle
import ast
from torchsummary import summary
        
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision import datasets, transforms


from utils.conv_type import FixedSubnetConv, SampleSubnetConv, GetGlobalSubnet, SubnetConv, GlobalSubnetConv  # <=== Add SubnetConv and GlobalSubnetConv here
from utils.logging import AverageMeter, ProgressMeter
from utils.net_utils import (
    set_model_prune_rate,
    bn_weight_init,
    freeze_model_weights,
    save_checkpoint,
    get_params,
    get_lr,
    LabelSmoothing,
)
from utils.schedulers import get_policy
from utils.conv_type import GetGlobalSubnet


from args import args
import importlib
from torch.utils.data import DataLoader, Subset

import data
import pandas as pd
import models

import numpy as np

#iiiiii
def load_validation_set(train_loader, val_indices_filepath="val_indices.pkl"):
    with open(val_indices_filepath, "rb") as f:
        val_indices = pickle.load(f)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    val_loader = torch.utils.data.DataLoader(
        train_loader.dataset,
        batch_size=train_loader.batch_size,
        sampler=val_sampler,
        num_workers=train_loader.num_workers,
        pin_memory=True,
    )
    return val_loader


def get_reproducible_train_subset(train_loader, subset_size=2000, seed=42):
    np.random.seed(seed)
    indices = np.random.choice(len(train_loader.dataset), subset_size, replace=False)
    subset_sampler = torch.utils.data.SubsetRandomSampler(indices)
    subset_loader = torch.utils.data.DataLoader(
        train_loader.dataset,
        batch_size=train_loader.batch_size,
        sampler=subset_sampler,
        num_workers=train_loader.num_workers,
        pin_memory=True,
    )
    return subset_loader



# Step 1: Load the pre-trained model
def load_model(checkpoint_path, model_class):
    print(f"Loading model from {checkpoint_path}")
    model = model_class()
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['state_dict']

    # Remove 'module.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # remove 'module.' prefix
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    print("Model loaded with following state dict keys:")
    print(new_state_dict.keys())
    return model

def main():
    print(args)
    torch.autograd.set_detect_anomaly(True)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        # Set to make training deterministic for seed
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    args.distributed = False

    if args.alphas is not None: #kkkkkkkkkk
            alphas_betas = [list(map(float, alpha_group.split(','))) for alpha_group in args.alphas.split(';')] #kkkkkkkkkk
            if all(len(ab) == len(alphas_betas[0]) for ab in alphas_betas): #kkkkkkkkkk
                if len(alphas_betas[0]) % 2 == 0: #kkkkkkkkkk
                    args.alphas_betas = [(alphas_betas[i][::2], alphas_betas[i][1::2]) for i in range(len(alphas_betas))]  # Separate alphas and betas #kkkkkkkkkk
                else: #kkkkkkkkkk
                    args.alphas = alphas_betas # Only alphas case #kkkkkkkkkk
            else: #kkkkkkkkkk
                raise ValueError("All alpha-beta pairs must have the same length") #kkkkkkkkkk

    
    # Simply call main_worker function
    main_worker(args)


def main_worker(args):
    args.gpu = None
    train, validate, modifier = get_trainer(args)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # Initialize the second model
    model2 = get_model(args)
    print("Second model architecture created")

    # Load weights and scores for model2 from the checkpoint
    checkpoint_path = '/content/drive/MyDrive/Colab Models/model_best (5).pth'
    if os.path.isfile(checkpoint_path):
        print(f"=> Loading checkpoint for model2 from '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint['state_dict']
        # Remove 'module.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # remove 'module.' prefix
            else:
                new_state_dict[k] = v

        model2.load_state_dict(new_state_dict, strict=False)
        print("Loaded checkpoint for model2")
    else:
        print(f"=> No checkpoint found at '{checkpoint_path}'")

    # Move the second model to GPU
    model2 = set_gpu(args, model2)
    print("Second model moved to GPU")

    data, train_augmentation = get_dataset(args)

    # Load validation set
    val_loader = load_validation_set(data.train_loader)

    if args.label_smoothing is None:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = LabelSmoothing(smoothing=args.label_smoothing)


    # Evaluate model2 on the subset of train set for each set of alphas #kkkkkkkkkk
    acc_list = [] #kkkkkkkkkk
    if hasattr(args, 'alphas_betas'): #kkkkkkkkkk
        for alphas, betas in args.alphas_betas: #kkkkkkkkkk
            set_alphas_betas(model2, alphas, betas) #kkkkkkkkkk
            acc1, acc5 = validate(val_loader, model2, criterion, args, writer=None, epoch=args.start_epoch)
            acc_list.append(acc1)
    elif hasattr(args, 'alphas'): #kkkkkkkkkk
        for alphas in args.alphas: #kkkkkkkkkk
            set_alphas(model2, alphas) #kkkkkkkkkk
            acc1, acc5 = validate(val_loader, model2, criterion, args, writer=None, epoch=args.start_epoch)
            acc_list.append(acc1)
    print(f"Subset Accuracy: {acc_list}") #kkkkkkkkkk


    # Save the accuracies to a CSV file
    results_path = '/content/drive/MyDrive/Colab_Results/biprop results/Results_vgg_small.csv'
    df = pd.read_csv(results_path)
    df['val_acc'] = acc_list
    df.to_csv(results_path, index=False)

def set_alphas(model, alphas): #kkkkkkkkkk
    alpha_idx = 0 #kkkkkkkkkk
    for name, module in model.named_modules(): #kkkkkkkkkk
        if isinstance(module, (SubnetConv, GlobalSubnetConv)): #kkkkkkkkkk
            module.alpha = alphas[alpha_idx] #kkkkkkkkkk
            alpha_idx += 1 #kkkkkkkkkk



            
def get_trainer(args):
    print(f"=> Using trainer from trainers.{args.trainer}")
    trainer = importlib.import_module(f"trainers.{args.trainer}")

    return trainer.train, trainer.validate, trainer.modifier


def set_gpu(args, model):
    assert torch.cuda.is_available(), "CPU-only experiments currently unsupported"

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    elif args.multigpu is None:
        device = torch.device("cpu")
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        print(f"=> Parallelizing on {args.multigpu} gpus")
        torch.cuda.set_device(args.multigpu[0])
        args.gpu = args.multigpu[0]
        model = torch.nn.DataParallel(model, device_ids=args.multigpu).cuda(
            args.multigpu[0]
        )
    if args.seed is None:
        cudnn.benchmark = True

    return model


def resume(args, model, optimizer):
    if os.path.isfile(args.resume):
        print(f"=> Loading checkpoint '{args.resume}'")

        checkpoint = torch.load(args.resume, map_location=f"cuda:{args.multigpu[0]}")
        if args.start_epoch is None:
            print(f"=> Setting new start epoch at {checkpoint['epoch']}")
            args.start_epoch = checkpoint["epoch"]

        best_acc1 = checkpoint["best_acc1"]

        model.load_state_dict(checkpoint["state_dict"])

        optimizer.load_state_dict(checkpoint["optimizer"])

        print(f"=> Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")

        return best_acc1
    else:
        print(f"=> No checkpoint found at '{args.resume}'")


def get_dataset(args):
    train_augmentation = 'Default'
    # Check if gaussian augmenation is being used
    if args.gaussian_aug:
      # Add _gaussian to args.set
      args.set = args.set + '_gaussian'
      # Set train augmentation to gaussian for logging purposes
      train_augmentation = 'Gaussian'
    # Check if augmix is being used
    elif args.augmix:
      # Add _augmix to args.set
      args.set = args.set + '_augmix'
      # Set train augmentation to augmix for logging purposes
      train_augmentation = 'Augmix'
    print(f"=> Getting {args.set} dataset")
    dataset = getattr(data, args.set)(args)

    return dataset, train_augmentation


def get_model(args):
    if args.first_layer_dense:
        args.first_layer_type = "DenseConv"

    # print("=> Creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()

    # applying sparsity to the network
    if (
        args.conv_type != "DenseConv"
        and args.conv_type != "SampleSubnetConv"
        and args.conv_type != "ContinuousSparseConv"
    ):
        if args.prune_rate < 0:
            raise ValueError("Need to set a positive prune rate")

        set_model_prune_rate(model, prune_rate=args.prune_rate)
        print(
            f"=> Rough estimate model params {sum(int(p.numel() * (1-args.prune_rate)) for n, p in model.named_parameters() if not n.endswith('scores'))}"
        )
    if args.bn_weight_init is not None or args.bn_bias_init is not None:
        bn_weight_init(model, weight=args.bn_weight_init, bias=args.bn_bias_init)

    # freezing the weights if we are only doing subnet training
    if args.freeze_weights:
        freeze_model_weights(model)

    return model


def _run_dir_exists(run_base_dir):
    log_base_dir = run_base_dir / "logs"
    ckpt_base_dir = run_base_dir / "checkpoints"

    return log_base_dir.exists() or ckpt_base_dir.exists()


def get_directories(args):
    if args.config is None or args.name is None:
        raise ValueError("Must have name and config")

    config = pathlib.Path(args.config).stem
    if args.log_dir is None:
        run_base_dir = pathlib.Path(
            f"runs/{config}/{args.name}/prune_rate={args.prune_rate}"
        )
    else:
        run_base_dir = pathlib.Path(
            f"{args.log_dir}/{config}/{args.name}/prune_rate={args.prune_rate}"
        )
    if args.width_mult != 1.0:
        run_base_dir = run_base_dir / "width_mult={}".format(str(args.width_mult))

    if _run_dir_exists(run_base_dir):
        rep_count = 0
        while _run_dir_exists(run_base_dir / str(rep_count)):
            rep_count += 1

        run_base_dir = run_base_dir / str(rep_count)

    log_base_dir = run_base_dir / "logs"
    ckpt_base_dir = run_base_dir / "checkpoints"

    if not run_base_dir.exists():
        os.makedirs(run_base_dir)

    (run_base_dir / "settings.txt").write_text(str(args))

    return run_base_dir, ckpt_base_dir, log_base_dir


def write_result_to_csv(**kwargs):
    results = pathlib.Path("runs") / "indiv_results4.csv"
    if args.results is not None:
        results = pathlib.Path(args.results) 

    if not results.exists():
        results.write_text(
            "Date Finished, "
            "Base Config, "
            "Current Val Top 1, "
            "Current Val Top 5, "
            "Best Val Top 1, "
            "Best Val Top 5, "
            "Best Train Top 1, "
            "Best Train Top 5, "
            "Name, "
            "Seed, "
            "Prune Rate, "
            "Learning Rate, "
            "Epochs, "
            "Weight Decay, "
            "Learn BN, "
            "Tune BN, "
            "Bias Only, "
            "Run Directory\n"
        )

    now = time.strftime("%m-%d-%y_%H:%M:%S")

    with open(results, "a+") as f:
        f.write(
            (
                "{now}, "
                "{base_config}, "
                "{curr_acc1:.02f}, "
                "{curr_acc5:.02f}, "
                "{best_acc1:.02f}, "
                "{best_acc5:.02f}, "
                "{best_train_acc1:.02f}, "
                "{best_train_acc5:.02f}, "
                "{name}, "
                "{seed}, "
                "{prune_rate}, "
                "{lr}, "
                "{epochs}, "
                "{weight_decay}, "
                "{learn_bn}, "
                "{tune_bn}, "
                "{bias_only}, "
                "{run_base_dir}\n"
            ).format(now=now, **kwargs)
        )

# Compute global prune rate at end of training
def global_prune_rate(model, args):
    # Initialize dictionary to store prune rates for each layer
    prune_dict = {}
    # Print breakdown of prune rate by layer
    print("\n==> Final layerwise prune rates in network:")
    # Loop over all model parameters and compute percentage of weights pruned globally
    total_weights = 0
    unpruned_weights = 0
    # Loop over all model parameters to get sparsity of each layer
    for n, m in model.named_modules():
      # Only add parameters that have prune_threshold as attribute
      if hasattr(m,'prune_threshold'):
        tmp_scores = m.clamped_scores.clone().detach()
        # Add to total_weights
        layer_total = int(torch.numel(tmp_scores))
        #print("Total number of weights in layer = ", t)
        total_weights += layer_total
        # Compute layer prune rate (doesn't seem to be stored correctly during multigpu runs)
        w = GetGlobalSubnet.apply(tmp_scores, m.weight.detach().clone(), m.prune_threshold)
        # Compute number of unpruned weights in layer
        layer_unpruned = torch.count_nonzero(w).item()
        # Compute pruning rate for current layer
        layer_prune_rate = 1 - (layer_unpruned/layer_total)
        # Compute number of pruned weights
        print("%s prune percentage: %lg" %(n,100*layer_prune_rate))
        unpruned_weights += layer_unpruned
        # Add prune_rate for current layer to dictionary
        prune_dict[n] = 100*layer_prune_rate
        # Set prune threshold value (same for all layers)
        pr_thresh = m.prune_threshold

    # Compute global pruning percentage
    #print ("total_weights = ", total_weights)
    #print ("pruned_weights = ", unpruned_weights)
    final_prune_rate = (1 - (unpruned_weights/total_weights))
    #print("Global pruning percentage: ", 100 * final_prune_rate)
    print("\n==> Global prune rate: ", 1-final_prune_rate)

    #print("\n==> Final prune threshold value: ", pr_thresh)

    # Return global prune rate
    return (1-final_prune_rate), prune_dict




if __name__ == "__main__":
    main()
