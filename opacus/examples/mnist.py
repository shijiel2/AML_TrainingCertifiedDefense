#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Runs MNIST training with differential privacy.

"""

from pathlib import Path
from tqdm import tqdm
from torchvision import datasets, transforms
from opacus.utils.uniform_sampler import UniformWithReplacementSampler, FixedSizedUniformWithReplacementSampler
from opacus import PrivacyEngine
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import logging
import socket
import argparse
import sys
sys.path.append("../..")
from notification import NOTIFIER

# Precomputed characteristics of the MNIST dataset
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


class SampleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, 2, padding=3)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(32 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        # x of shape [B, 1, 28, 28]
        x = F.relu(self.conv1(x))  # -> [B, 16, 14, 14]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 16, 13, 13]
        x = F.relu(self.conv2(x))  # -> [B, 32, 5, 5]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 32, 4, 4]
        x = x.view(-1, 32 * 4 * 4)  # -> [B, 512]
        x = F.relu(self.fc1(x))  # -> [B, 32]
        x = self.fc2(x)  # -> [B, 10]
        return x

    def name(self):
        return "SampleConvNet"


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []
    for _batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # if not args.disable_dp:
    #     epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(args.delta)
    #     print(
    #         f"Train Epoch: {epoch} \t"
    #         f"Loss: {np.mean(losses):.6f} "
    #         f"(ε = {epsilon:.2f}, δ = {args.delta}) for α = {best_alpha}"
    #     )
    # else:
    #     print(f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f}")
    logging.info(f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f}")


def test(args, model, device, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    logging.info(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    return correct / len(test_loader.dataset)


def pred(args, model, device, test_dataset):
    model.eval()

    X, y = next(iter(torch.utils.data.DataLoader(
        test_dataset, batch_size=len(test_dataset))))
    X, y = X.to(device), y.to(device)

    y_pred = model(X).max(1)[1]

    return y_pred


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "-sr",
        "--sample-rate",
        type=float,
        default=0.001,
        metavar="SR",
        help="sample rate used for batch construction (default: 0.001)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1024,
        metavar="TB",
        help="input batch size for testing (default: 1024)",
    )
    parser.add_argument(
        "-n",
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 14)",
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
        "--lr",
        type=float,
        default=0.1,
        metavar="LR",
        help="learning rate (default: .1)",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        metavar="S",
        help="Noise multiplier (default 1.0)",
    )
    parser.add_argument(
        "-c",
        "--max-per-sample-grad_norm",
        type=float,
        default=1.0,
        metavar="C",
        help="Clip per-sample gradients to this norm (default 1.0)",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        metavar="D",
        help="Target delta (default: 1e-5)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="GPU ID for this process (default: 'cuda')",
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
        "--data-root",
        type=str,
        default="../mnist",
        help="Where MNIST is/will be stored",
    )
    parser.add_argument(
        "--results-folder",
        type=str,
        default="../results/mnist",
        help="Where MNIST results is/will be stored",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="SampleConvNet",
        help="Name of the model",
    )
    parser.add_argument(
        "--sub-training-size",
        type=int,
        default=0,
        help="Size of bagging",
    )
    parser.add_argument(
        "--load-model",
        action="store_true",
        default=False,
        help="Load model not train (default: false)",
    )
    parser.add_argument(
        "--sub-acc-test",
        action="store_true",
        default=False,
        help="Test subset V.S. acc (default: false)",
    )
    parser.add_argument(
        "--train-mode",
        type=str,
        default="DP",
        help="Mode of training: DP, Bagging, Sub-DP",
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    kwargs = {"num_workers": 1, "pin_memory": True}

    def gen_sub_dataset(dataset, sub_training_size, with_replacement):
        indexs = np.random.choice(len(dataset), sub_training_size, replace=with_replacement)
        dataset = torch.utils.data.Subset(dataset, indexs)
        print(f"Sub-dataset size {len(dataset)}")
        return dataset

    def gen_train_dataset_loader(or_sub_training_size=None):
        train_dataset = datasets.MNIST(
            args.data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
                ]
            ),
        )

        # Generate sub-training dataset if necessary
        sub_training_size = args.sub_training_size if or_sub_training_size is None else or_sub_training_size
        if args.train_mode == 'Bagging' or args.train_mode == 'Sub-DP':
            train_dataset = gen_sub_dataset(train_dataset, sub_training_size, True)

        if args.train_mode == 'DP' or args.train_mode == 'Sub-DP':
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                generator=None,
                batch_sampler=UniformWithReplacementSampler(
                    num_samples=len(train_dataset),
                    sample_rate=args.sample_rate,
                    generator=None,
                ),
                **kwargs,
            )
        else:
            print('No Gaussian Sampler')
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=32,
                shuffle=True,
                **kwargs,
            )
        return train_dataset, train_loader

    def gen_test_dataset_loader():
        test_dataset = datasets.MNIST(
            args.data_root,
            train=False,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
                ]
            ),
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.test_batch_size,
            shuffle=True,
            **kwargs,
        )

        return test_dataset, test_loader

    # folder for this experiment
    if args.train_mode == 'DP':
        result_folder = (
            f"{args.results_folder}/{args.model_name}_{args.lr}_{args.sigma}_"
            f"{args.max_per_sample_grad_norm}_{args.sample_rate}_{args.epochs}_{args.n_runs}"
        )
    elif args.train_mode == 'Bagging':
        result_folder = (
            f"{args.results_folder}/Bagging_{args.model_name}_{args.lr}_{args.sub_training_size}_"
            f"{args.epochs}_{args.n_runs}"
        )
    elif args.train_mode == 'Sub-DP':
        result_folder = (
            f"{args.results_folder}/{args.model_name}_{args.lr}_{args.sigma}_"
            f"{args.max_per_sample_grad_norm}_{args.sample_rate}_{args.epochs}_{args.sub_training_size}_{args.n_runs}"
        )
    print(f'Result folder: {result_folder}')
    models_folder = f"{result_folder}/models"
    Path(models_folder).mkdir(parents=True, exist_ok=True)

    # log file for this experiment
    logging.basicConfig(
        filename=f"{result_folder}/train.log", filemode='w', level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())

    # collect votes from all models
    test_dataset, test_loader = gen_test_dataset_loader()
    aggregate_result = np.zeros([len(test_dataset), 10 + 1], dtype=np.int)
    acc_list = []
    
    # # use this code for "sub_training_size V.S. acc"
    if args.sub_acc_test:
        sub_acc_list = []

    for run_idx in range(args.n_runs):
        # pre-training stuff
        if args.model_name == 'SampleConvNet':
            model = SampleConvNet().to(device)
        else:
            logging.warn(f"Model name {args.model_name} invaild.")
            exit()
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0)
        if args.train_mode in ['DP', 'Sub-DP']:
            privacy_engine = PrivacyEngine(
                model,
                sample_rate=args.sample_rate,
                alphas=[
                    1 + x / 10.0 for x in range(1, 100)] + list(range(12, 1500)),
                noise_multiplier=args.sigma,
                max_grad_norm=args.max_per_sample_grad_norm,
            )
            privacy_engine.attach(optimizer)

        # training
        if args.load_model:
            model.load_state_dict(torch.load(
                f"{models_folder}/model_{run_idx}.pt"))
        else:
            # use this code for "sub_training_size V.S. acc"
            if args.sub_acc_test:
                sub_training_size = int(60000 - 60000 / args.n_runs * run_idx)
                _, train_loader = gen_train_dataset_loader(sub_training_size)
            else:
                _, train_loader = gen_train_dataset_loader()
            epoch_acc_epsilon = []
            for epoch in range(1, args.epochs + 1):
                train(args, model, device, train_loader, optimizer, epoch)
                if args.run_test:
                    test(args, model, device, test_loader)
                if run_idx == 0:
                    acc = test(args, model, device, test_loader)
                    if args.train_mode in ['DP', 'Sub-DP']:
                        eps, _ = optimizer.privacy_engine.get_privacy_spent(args.delta)
                        epoch_acc_epsilon.append((acc, eps))    
            if run_idx == 0:
                np.save(f"{result_folder}/epoch_acc_eps", epoch_acc_epsilon)
        
        # use this code for "sub_training_size V.S. acc"
        if args.sub_acc_test:
            sub_acc_list.append((sub_training_size, test(args, model, device, test_loader)))

        # post-training stuff
        if run_idx == 0 and args.train_mode in ['DP', 'Sub-DP']:
            rdp_alphas, rdp_epsilons = optimizer.privacy_engine.get_rdp_privacy_spent()
            dp_epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(
                args.delta)
            rdp_steps = optimizer.privacy_engine.steps
            logging.info(
                f"epsilon {dp_epsilon}, best_alpha {best_alpha}, steps {rdp_steps}")
            print(
                f"epsilon {dp_epsilon}, best_alpha {best_alpha}, steps {rdp_steps}")
            if args.save_model:
                np.save(f"{result_folder}/rdp_epsilons", rdp_epsilons)
                np.save(f"{result_folder}/rdp_alphas", rdp_alphas)
                np.save(f"{result_folder}/rdp_steps", rdp_steps)
                np.save(f"{result_folder}/dp_epsilon", dp_epsilon)
        # save preds
        aggregate_result[np.arange(0, len(test_dataset)), pred(
            args, model, device, test_dataset).cpu()] += 1
        acc_list.append(test(args, model, device, test_loader))
        # save model
        if not args.load_model and args.save_model:
            torch.save(model.state_dict(),
                       f"{models_folder}/model_{run_idx}.pt")
    # finish trining all models, save results
    aggregate_result[np.arange(0, len(test_dataset)), -1] = next(
        iter(torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset))))[1]
    np.save(f"{result_folder}/aggregate_result", aggregate_result)
    np.save(f"{result_folder}/acc_list", acc_list)

    # use this code for "sub_training_size V.S. acc"
    if args.sub_acc_test:
        np.save(f"{result_folder}/subset_acc_list", sub_acc_list)


if __name__ == "__main__":
    main()
    # NOTIFIER.notify(socket.gethostname(), 'Screen Job pyvacy, Done.')
