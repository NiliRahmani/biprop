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
import models

import numpy as np

import numpy as np
import torch
import pickle

def load_used_indices(filepath):
    try:
        with open(filepath, "rb") as f:
            return set(pickle.load(f))
    except FileNotFoundError:
        return set()

def save_used_indices(indices, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(indices, f)

def reset_used_indices(filepath):
    with open(filepath, "wb") as f:
        pickle.dump(set(), f)

# def get_reproducible_train_subset(train_loader, subset_size=2000, seed=42, used_indices_filepath="used_indices.pkl", val_indices_filepath="val_indices.pkl"):
#     np.random.seed(seed)
#     total_indices = set(range(len(train_loader.dataset)))
#     used_indices = load_used_indices(used_indices_filepath)
    
#     with open(val_indices_filepath, "rb") as f:
#         val_indices = set(pickle.load(f))
    
#     available_indices = list(total_indices - used_indices - val_indices)
    
#     print(f"Total dataset size: {len(total_indices)}")
#     print(f"Used indices so far: {len(used_indices)}")
#     print(f"Available indices: {len(available_indices)}")

#     if len(available_indices) < subset_size:
#         print("Not enough available indices, resetting used indices.")
#         reset_used_indices(used_indices_filepath)
#         used_indices = set()
#         available_indices = list(total_indices - val_indices)

#     new_indices = np.random.choice(available_indices, subset_size, replace=False)
#     used_indices.update(new_indices)
#     save_used_indices(used_indices, used_indices_filepath)

#     print(f"New subset indices: {new_indices[:10]}...")  # Print the first 10 new indices for verification

#     subset_sampler = torch.utils.data.SubsetRandomSampler(new_indices)
    
#     subset_loader = torch.utils.data.DataLoader(
#         train_loader.dataset,
#         batch_size=train_loader.batch_size,
#         sampler=subset_sampler,
#         num_workers=train_loader.num_workers,
#         pin_memory=True,
#     )
#     return subset_loader

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

    # Parse alphas
    if args.alphas is not None:
        args.alphas = [list(map(float, alpha_group.split(','))) for alpha_group in args.alphas.split(';')]

    if args.betas is not None:  # lllllllllllll
        args.betas = [list(map(float, beta_group.split(','))) for beta_group in args.betas.split(';')]  # lllllllllllll

    
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

    data, train_augmentation = get_dataset(args)  ##data: 5k samples out of train set

    if args.label_smoothing is None:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = LabelSmoothing(smoothing=args.label_smoothing)

    # Create the reproducible subset of train data
    train_loader = data.train_loader
    # subset_loader = get_reproducible_train_subset(train_loader, subset_size=2000, seed=args.seed, used_indices_filepath="used_indices.pkl", val_indices_filepath="/content/biprop/val_indices.pkl")

    # # Evaluate model2 on the subset of train set for each set of alphas
    # acc_list = []
    # for alphas in args.alphas:
    #     set_alphas(model2, alphas)
    #     acc1_2, acc5_2 = validate(subset_loader, model2, criterion, args, writer=None, epoch=args.start_epoch)
    #     acc_list.append(acc1_2)
    # print(f"Subset Accuracy: {acc_list}")

    # Evaluate model2 on the subset of train set for each set of alphas (and betas if provided)
    acc_list = []
    if args.betas is not None:  # lllllllllllll
        for alphas, betas in zip(args.alphas, args.betas):  # lllllllllllll
            set_alphas_betas(model2, alphas, betas)  # lllllllllllll
            acc1_2, acc5_2 = validate(train_loader, model2, criterion, args, writer=None, epoch=args.start_epoch)
            acc_list.append(acc1_2)
    else:  # lllllllllllll
        for alphas in args.alphas:  # lllllllllllll
            set_alphas_betas(model2, alphas)  # lllllllllllll
            acc1_2, acc5_2 = validate(train_loader, model2, criterion, args, writer=None, epoch=args.start_epoch)
            acc_list.append(acc1_2)
    print(f"Subset Accuracy: {acc_list}")



# def set_alphas(model, alphas):
#     alpha_idx = 0
#     for name, module in model.named_modules():
#         if isinstance(module, (SubnetConv, GlobalSubnetConv)):
#             module.alpha = alphas[alpha_idx]
#             alpha_idx += 1
def set_alphas_betas(model, alphas, betas=None):  # lllllllllllll
    alpha_idx = 0
    for name, module in model.named_modules():
        if isinstance(module, (SubnetConv, GlobalSubnetConv)):
            module.alpha = alphas[alpha_idx]  # lllllllllllll
            if betas is not None:  # lllllllllllll
                module.beta = betas[alpha_idx]  # lllllllllllll
            alpha_idx += 1


            
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
    
    # Check if gaussian augmentation is being used
    if args.gaussian_aug:
        args.set = args.set + '_gaussian'
        train_augmentation = 'Gaussian'
    elif args.augmix:
        args.set = args.set + '_augmix'
        train_augmentation = 'Augmix'

    print(f"=> Getting {args.set} dataset")

    # Define the transformations (same as those in data module)
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]),
    ])

    # Load the CIFAR10 dataset
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    # Ensure reproducibility by setting the random seed
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Total number of samples in the dataset
    total_samples = 50000 #len(dataset)

    # Create a subset with 45,000 random samples
    num_samples1 = 45000
    all_indices = list(range(total_samples))
    indices1 = random.sample(all_indices, num_samples1)

    # Create the remaining subset with 5,000 samples
    indices2 = list(set(all_indices) - set(indices1))

    # Subset objects for both datasets
    subset1 = Subset(dataset, indices1)
    subset2 = Subset(dataset, indices2)

    # Split subset1 into a training set with 40,000 samples and a validation set with 5,000 samples
    num_val_samples = 5000
    num_train_samples = num_samples1 - num_val_samples
    subset1_train, subset1_val = random_split(subset1, [num_train_samples, num_val_samples])

    # Create DataLoaders for both subsets
    train_loader1 = DataLoader(subset1_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(subset1_val, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    train_loader2 = DataLoader(subset2, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    # Wrap the DataLoaders in a class to match the expected structure
    class DatasetWrapper:
        def __init__(self, train_loader, val_loader=None):
            self.train_loader = train_loader
            self.val_loader = val_loader

    # Return dataset1 with both train_loader and val_loader, and dataset2 with only train_loader
    return DatasetWrapper(train_loader2), train_augmentation

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
