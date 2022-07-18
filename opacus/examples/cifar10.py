#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Runs CIFAR10 training with differential privacy.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
# import torch.utils.tensorboard as tensorboard
import torchvision.transforms as transforms
import torchvision.models as models
from opacus import PrivacyEngine
from opacus.layers import DifferentiallyPrivateDistributedDataParallel as DPDDP
# from opacus.utils import stats
from opacus.utils.uniform_sampler import UniformWithReplacementSampler, FixedSizedUniformWithReplacementSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.datasets import CIFAR10
from tqdm import tqdm
from models import ResNet18
from certify_utilis import result_folder_path_generator
from opacus.utils import module_modification
from torch.utils.data.distributed import DistributedSampler



# def setup():
#     if sys.platform == "win32":
#         raise NotImplementedError("Windows version of multi-GPU is not supported yet.")
#     else:
#         # initialize the process group
#         torch.distributed.init_process_group(
#             init_method="env://",
#             backend="nccl",
#         )

def setup(args):

    if torch.cuda.device_count() > 1:
        torch.distributed.init_process_group(backend="nccl")
        local_rank = torch.distributed.get_rank()
        rank = 0
        world_size = torch.cuda.device_count()
        return (rank, local_rank, world_size)

    else:
        logging.info(f"Running on a single GPU.")
        return (0, 0, 1)

def cleanup():
    torch.distributed.destroy_process_group()


def convnet(num_classes):
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(start_dim=1, end_dim=-1),
        nn.Linear(128, num_classes, bias=True),
    )


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = func.relu(self.conv1(x))
        x = func.max_pool2d(x, 2)
        x = func.relu(self.conv2(x))
        x = func.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def accuracy(preds, labels):
    return (preds == labels).mean()


def train(args, model, train_loader, optimizer, epoch, device, batch_num=None):
    model.train()
    criterion = nn.CrossEntropyLoss()

    losses = []
    top1_acc = []

    for i, (images, target) in enumerate(train_loader):

        images = images.to(device)
        target = target.to(device)

        # compute output
        output = model(images)
        loss = criterion(output, target)
        preds = np.argmax(output.detach().cpu().numpy(), axis=1)
        labels = target.detach().cpu().numpy()

        # measure accuracy and record loss
        acc1 = accuracy(preds, labels)

        losses.append(loss.item())
        top1_acc.append(acc1)
        # stats.update(stats.StatType.TRAIN, acc1=acc1)

        # compute gradient and do SGD step
        loss.backward()

        # make sure we take a step after processing the last mini-batch in the
        # epoch to ensure we start the next epoch with a clean state
        if ((i + 1) % args.n_accumulation_steps == 0) or ((i + 1) == len(train_loader)):
            optimizer.step()
            optimizer.zero_grad()
        else:
            optimizer.virtual_step()

        if batch_num is not None and i >= batch_num:
            break

        # if i % args.print_freq == 0:
        #     if not args.disable_dp:
        #         epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(
        #             args.delta
        #         )
        #         print(
        #             f"\tTrain Epoch: {epoch} \t"
        #             f"Loss: {np.mean(losses):.6f} "
        #             f"Acc@1: {np.mean(top1_acc):.6f} "
        #             f"(ε = {epsilon:.2f}, δ = {args.delta}) for α = {best_alpha}"
        #         )
        #     else:
        #         print(
        #             f"\tTrain Epoch: {epoch} \t"
        #             f"Loss: {np.mean(losses):.6f} "
        #             f"Acc@1: {np.mean(top1_acc):.6f} "
        #         )


def test(args, model, test_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    top1_acc = []

    with torch.no_grad():
        for images, target in test_loader:
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc1 = accuracy(preds, labels)

            losses.append(loss.item())
            top1_acc.append(acc1)

    top1_avg = np.mean(top1_acc)
    # stats.update(stats.StatType.TEST, acc1=top1_avg)

    strv = f"\tTest set(size:{len(test_loader.dataset)}, device:{device}):" f"Loss: {np.mean(losses):.6f} " f"Acc@1: {top1_avg :.6f} "
    logging.info(strv)
    return top1_avg


def pred(args, model, test_loader, device):
    model.eval()
    preds_list = []
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            output = model(images)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            preds_list.extend(preds)
    return np.array(preds_list)

    # model.eval()
    # X, y = next(iter(torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset))))
    # X, y  = X.to(device), y.to(device)
    # y_pred = model(X).max(1)[1]
    # return y_pred


def softmax(args, model, test_loader, device):
    model.eval()
    softmax_list = []
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            output = model(images)
            softmax = nn.Softmax(dim=1)(output).detach().cpu().numpy()
            softmax_list.extend(softmax)
    return np.array(softmax_list)
    
    # model.eval()
    # X, y = next(iter(torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset))))
    # X, y  = X.to(device), y.to(device)
    # softmax = nn.Softmax(dim=1)(model(X))
    # return softmax


# flake8: noqa: C901
def main():
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 DP Training")
    parser.add_argument(
        "-j",
        "--workers",
        default=0,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 2)",
    )
    parser.add_argument(
        "--epochs",
        default=90,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--start-epoch",
        default=1,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument(
        "-b",
        "--batch-size-test",
        default=256,
        type=int,
        metavar="N",
        help="mini-batch size for test dataset (default: 256), this is the total "
        "batch size of all GPUs on the current node when "
        "using Data Parallel or Distributed Data Parallel",
    )
    parser.add_argument(
        "--sample-rate",
        default=0.04,
        type=float,
        metavar="SR",
        help="sample rate used for batch construction (default: 0.005)",
    )
    parser.add_argument(
        "-na",
        "--n_accumulation_steps",
        default=1,
        type=int,
        metavar="N",
        help="number of mini-batches to accumulate into an effective batch",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.1,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="SGD momentum"
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=5e-4,
        type=float,
        metavar="W",
        help="SGD weight decay",
        dest="weight_decay",
    )
    parser.add_argument(
        "-p",
        "--print-freq",
        default=10,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on validation set",
    )
    parser.add_argument(
        "--seed", default=None, type=int, help="seed for initializing training. "
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="GPU ID for this process (default: 'cuda')",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.5,
        metavar="S",
        help="Noise multiplier (default 1.0)",
    )
    parser.add_argument(
        "-c",
        "--max-per-sample-grad_norm",
        type=float,
        default=10.0,
        metavar="C",
        help="Clip per-sample gradients to this norm (default 1.0)",
    )
    parser.add_argument(
        "--secure-rng",
        action="store_true",
        default=False,
        help="Enable Secure RNG to have trustworthy privacy guarantees. Comes at a performance cost",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        metavar="D",
        help="Target delta (default: 1e-5)",
    )

    parser.add_argument(
        "--checkpoint-file",
        type=str,
        default="checkpoint",
        help="path to save check points",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="../cifar10",
        help="Where CIFAR10 is/will be stored",
    )
    parser.add_argument(
        "--log-dir", type=str, default="", help="Where Tensorboard log will be stored"
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="SGD",
        help="Optimizer to use (Adam, RMSprop, SGD)",
    )
    parser.add_argument(
        "--lr-schedule", type=str, choices=["constant", "cos"], default="cos"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank if multi-GPU training, -1 for single GPU training",
    )
    # New added args
    parser.add_argument(
        "--model-name",
        type=str,
        default="ConvNet",
        help="Name of the model structure",
    )
    parser.add_argument(
        "--results-folder",
        type=str,
        default="../results/cifar10",
        help="Where CIFAR10 results is/will be stored",
    )
    parser.add_argument(
        "--sub-training-size",
        type=int,
        default=0,
        help="Size of bagging",
    )
    parser.add_argument(
        "-r",
        "--n-runs",
        type=int,
        default=1,
        metavar="R",
        help="number of runs to average on (default: 1)",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="Save the trained model (default: false)",
    )
    parser.add_argument(
        "--run-test",
        action="store_true",
        default=False,
        help="Run test for the model (default: false)",
    )
    parser.add_argument(
        "--load-model",
        action="store_true",
        default=False,
        help="Load model not train (default: false)",
    )
    parser.add_argument(
        "--train-mode",
        type=str,
        default="DP",
        help="Train mode: DP, Sub-DP, Bagging",
    )
    parser.add_argument(
        "--sub-acc-test",
        action="store_true",
        default=False,
        help="Test subset V.S. acc (default: false)",
    )

    args = parser.parse_args()

    # folder path
    result_folder = result_folder_path_generator(args)
    print(f'Result folder: {result_folder}')
    models_folder = f"{result_folder}/models"
    Path(models_folder).mkdir(parents=True, exist_ok=True)

    # logging
    logging.basicConfig(filename=f"{result_folder}/train.log", filemode='w', level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())

    # distributed = False
    # if args.local_rank != -1:
    #     setup()
    #     distributed = True

    # Sets `world_size = 1` if you run on a single GPU with `args.local_rank = -1`
    if args.device != "cpu":
        rank, local_rank, world_size = setup(args)
        device = local_rank
    else:
        device = "cpu"
        rank = 0
        world_size = 1


    if args.train_mode == 'Bagging' and args.n_accumulation_steps > 1:
        raise ValueError("Virtual steps only works with enabled DP")

    # The following lines enable stat gathering for the clipping process
    # and set a default of per layer clipping for the Privacy Engine
    clipping = {"clip_per_layer": False, "enable_stat": True}

    generator = None

    augmentations = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]

    train_transform = transforms.Compose(
        augmentations + normalize if args.train_mode == 'Bagging' else normalize
    )
    test_transform = transforms.Compose(normalize)

    def gen_sub_dataset(dataset, sub_training_size, with_replacement):
        indexs = np.random.choice(len(dataset), sub_training_size, replace=with_replacement)
        dataset = torch.utils.data.Subset(dataset, indexs)
        print(f"Sub-dataset size {len(dataset)}")
        return dataset

    def gen_train_dataset_loader(sub_training_size):
        train_dataset = CIFAR10(
            root=args.data_root, train=True, download=True, transform=train_transform
        )

        if args.train_mode == 'Sub-DP' or args.train_mode == 'Bagging':
            train_dataset = gen_sub_dataset(train_dataset, sub_training_size, True)
        
        batch_num = None

        if world_size > 1:
            dist_sampler = DistributedSampler(train_dataset)
        else:
            dist_sampler = None

        # batch_size = int(args.sample_rate * len(train_dataset))

        # if args.train_mode == 'DP' or args.train_mode == 'Sub-DP':
        #     train_loader = torch.utils.data.DataLoader(
        #         train_dataset,
        #         num_workers=args.workers,
        #         generator=generator,
        #         batch_size=batch_size,
        #         sampler=dist_sampler,
        #     )
        # elif args.train_mode == 'Sub-DP-no-amp':
        #     train_loader = torch.utils.data.DataLoader(
        #         train_dataset,
        #         num_workers=args.workers,
        #         generator=generator,
        #         batch_size=batch_size,
        #         sampler=dist_sampler,
        #     )
        #     batch_num = int(sub_training_size / int(args.sample_rate * len(train_dataset)))
        # else:
        #     print('No Gaussian Sampler')
        #     train_loader = torch.utils.data.DataLoader(
        #         train_dataset,
        #         num_workers=args.workers,
        #         generator=generator,
        #         batch_size=int(128/world_size),
        #         sampler=dist_sampler,
        #     )
        # return train_dataset, train_loader, batch_num
        
        if args.train_mode == 'DP' or args.train_mode == 'Sub-DP':
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                num_workers=args.workers,
                generator=generator,
                batch_sampler=FixedSizedUniformWithReplacementSampler(
                    num_samples=len(train_dataset),
                    sample_rate=args.sample_rate/world_size,
                    train_size=len(train_dataset)/world_size,
                    generator=generator,
                ),
            )
        elif args.train_mode == 'Sub-DP-no-amp':
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                num_workers=args.workers,
                generator=generator,
                batch_sampler=FixedSizedUniformWithReplacementSampler(
                    num_samples=len(train_dataset),
                    sample_rate=args.sample_rate/world_size,
                    train_size=sub_training_size/world_size,
                    generator=generator,
                ),
            )
        else:
            print('No Gaussian Sampler')
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                num_workers=args.workers,
                generator=generator,
                batch_size=int(256/world_size),
                sampler=dist_sampler,
            )
        return train_dataset, train_loader, batch_num

    def gen_test_dataset_loader():
        test_dataset = CIFAR10(
            root=args.data_root, train=False, download=True, transform=test_transform
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size_test,
            shuffle=False,
            num_workers=args.workers,
        )
        return test_dataset, test_loader
    

    # if distributed and args.device == "cuda":
    #     args.device = "cuda:" + str(args.local_rank)
    # device = torch.device(args.device)

    """ Here we go the training and testing process """
    
    # collect votes from all models
    test_dataset, test_loader = gen_test_dataset_loader()
    aggregate_result = np.zeros([len(test_dataset), 10 + 1], dtype=np.int)
    aggregate_result_softmax = np.zeros([args.n_runs, len(test_dataset), 10 + 1], dtype=np.float32)
    acc_list = []

    # use this code for "sub_training_size V.S. acc"
    if args.sub_acc_test:
        sub_acc_list = []
    
    for run_idx in range(args.n_runs):
        # Pre-training stuff for each base classifier
        
        # Define the model
        if args.model_name == 'ConvNet':
            model = convnet(num_classes=10)
        elif args.model_name == 'ResNet18-BN':
            model = ResNet18(10)
            # model = module_modification.convert_batchnorm_modules(models.resnet18(pretrained=False, num_classes=10))
            # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            # model = models.resnet18(pretrained=False, num_classes=10)
            # model = module_modification.convert_batchnorm_modules(ResNet18(10))
        elif args.model_name == 'ResNet18-GN':
            model = module_modification.convert_batchnorm_modules(ResNet18(10))
        elif args.model_name == 'LeNet':
            model = LeNet()
        else:
            exit(f'Model name {args.model_name} invaild.')
        model = model.to(device)
        
        if world_size > 1:
            if not args.train_mode == 'Bagging':
                model = DPDDP(model)
            else:
                # model = DDP(model, device_ids=[args.local_rank])
                model = DDP(model, device_ids=[device])
        
        # Define the optimizer
        if args.optim == "SGD":
            optimizer = optim.SGD(
                model.parameters(),
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
            )
        elif args.optim == "RMSprop":
            optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
        elif args.optim == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
        else:
            raise NotImplementedError("Optimizer not recognized. Please check spelling")

        # Define the DP engine
        if args.train_mode != 'Bagging':
            privacy_engine = PrivacyEngine(
                model,
                sample_rate=args.sample_rate * args.n_accumulation_steps / world_size,
                alphas=[1 + x / 10.0 for x in range(1, 100)],
                noise_multiplier= 0.0 if args.train_mode == 'Bagging' else args.sigma,
                max_grad_norm=args.max_per_sample_grad_norm,
                secure_rng=args.secure_rng,
                **clipping,
            )
            privacy_engine.attach(optimizer)
        
        # Training and testing
        model_pt_file = f"{models_folder}/model_{run_idx}.pt"
        def training_process():
            logging.info(f'training model_{run_idx}...')
            # use this code for "sub_training_size V.S. acc"
            if args.sub_acc_test:
                sub_training_size = int(50000 - 50000 / args.n_runs * run_idx)
                _, train_loader, batch_num = gen_train_dataset_loader(sub_training_size)
            else:    
                sub_training_size = args.sub_training_size
                _, train_loader, batch_num = gen_train_dataset_loader(sub_training_size)

            epoch_acc_epsilon = []
            for epoch in range(args.start_epoch, args.epochs + 1):
                if args.lr_schedule == "cos":
                    lr = args.lr * 0.5 * (1 + np.cos(np.pi * epoch / (args.epochs + 1)))
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr

                train(args, model, train_loader, optimizer, epoch, device, batch_num)

                # if args.run_test:
                #     logging.info(f'Epoch: {epoch}')
                #     test(args, model, test_loader, device)

                # if run_idx == 0:
                #     logging.info(f'Epoch: {epoch}')
                #     acc = test(args, model, test_loader, device)
                #     if args.train_mode in ['DP', 'Sub-DP', 'Sub-DP-no-amp']:
                #         eps, _ = optimizer.privacy_engine.get_privacy_spent(args.delta)
                #         epoch_acc_epsilon.append((acc, eps))
            
            
            if run_idx == 0:
                np.save(f"{result_folder}/epoch_acc_eps", epoch_acc_epsilon)
            
            # Post-training stuff 

            # use this code for "sub_training_size V.S. acc"
            if args.sub_acc_test:
                sub_acc_list.append((sub_training_size, test(args, model, test_loader, device)))

            # save the DP related data
            if run_idx == 0 and args.train_mode in ['DP', 'Sub-DP', 'Sub-DP-no-amp']:
                rdp_alphas, rdp_epsilons = optimizer.privacy_engine.get_rdp_privacy_spent()
                dp_epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(args.delta)
                rdp_steps = optimizer.privacy_engine.steps
                logging.info(f"epsilon {dp_epsilon}, best_alpha {best_alpha}, steps {rdp_steps}")

                logging.info(f"sample_rate {optimizer.privacy_engine.sample_rate}, noise_multiplier {optimizer.privacy_engine.noise_multiplier}, steps {optimizer.privacy_engine.steps}")
                
                np.save(f"{result_folder}/rdp_epsilons", rdp_epsilons)
                np.save(f"{result_folder}/rdp_alphas", rdp_alphas)
                np.save(f"{result_folder}/rdp_steps", rdp_steps)
                np.save(f"{result_folder}/dp_epsilon", dp_epsilon)
        
        if os.path.isfile(model_pt_file) or args.load_model:
            try:
                torch.distributed.barrier()
                logging.info(f'loading existing model_{run_idx}...')
                map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
                model.load_state_dict(torch.load(model_pt_file, map_location=map_location))
            except Exception as inst:
                logging.info(f'fail to load model with error: {inst}')
                training_process()
        else:
            training_process()
        
        # save preds and model
        aggregate_result[np.arange(0, len(test_dataset)), pred(args, model, test_loader, device)] += 1
        aggregate_result_softmax[run_idx, np.arange(0, len(test_dataset)), 0:10] = softmax(args, model, test_loader, device)
        acc_list.append(test(args, model, test_loader, device))
        if not args.load_model and args.save_model and local_rank == 0:
            torch.save(model.state_dict(), model_pt_file)

    # Finish trining all models, save results
    aggregate_result[np.arange(0, len(test_dataset)), -1] = next(iter(torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset))))[1]
    aggregate_result_softmax[:, np.arange(0, len(test_dataset)), -1] = next(iter(torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset))))[1]
    np.save(f"{result_folder}/aggregate_result", aggregate_result)
    np.save(f"{result_folder}/aggregate_result_softmax", aggregate_result_softmax)
    np.save(f"{result_folder}/acc_list", acc_list)

    # use this code for "sub_training_size V.S. acc"
    if args.sub_acc_test:
        np.save(f"{result_folder}/subset_acc_list", sub_acc_list)

    if world_size > 1:
        cleanup()


if __name__ == "__main__":
    main()
