from __future__ import print_function
import numpy as np
from statsmodels.stats.proportion import (
    proportion_confint,
    multinomial_proportions_confint,
)
import argparse
import math
import os
import logging

import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from opacus import PrivacyEngine
import matplotlib.pyplot as plt
import scipy.stats
from certify_utilis import *


parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, default="0.001")
parser.add_argument(
    "-sr",
    "--sample-rate",
    type=float,
    default=0.001,
    metavar="SR",
    help="sample rate used for batch construction (default: 0.001)",
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
    "--plot",
    action="store_true",
    default=False,
    help="plot the certified acc",
)
parser.add_argument(
    "--training-size",
    type=int,
    default=60000,
    help="Size of training set",
)
parser.add_argument(
    "--train-mode",
    type=str,
    default="DP",
    help="Name of the methods: DP, Sub-DP, Bagging",
)
parser.add_argument(
    "--radius-range",
    type=int,
    default=150,
    help="Size of training set",
)
parser.add_argument(
    "--sub-training-size",
    type=int,
    default=30000,
    help="Size of training set",
)
args = parser.parse_args()


def certified_acc_against_radius(certified_poisoning_size_array, radius_range=50):
    certified_radius_list = list(range(radius_range))
    certified_acc_list = []

    for radius in certified_radius_list:
        certified_acc_list.append(
            len(
                certified_poisoning_size_array[
                    np.where(certified_poisoning_size_array >= radius)
                ]
            )
            / float(num_data)
        )
    return certified_acc_list, certified_radius_list


def certified_acc_against_radius_dp_baseline(clean_acc_list, dp_epsilon, dp_delta=1e-5, radius_range=50):
    _, est_clean_acc, _ = single_ci(clean_acc_list, args.alpha)
    # est_clean_acc = sum(clean_acc_list) / len(clean_acc_list)
    c_bound = 1
    def dp_baseline_certified_acc(k):
        p1 = np.e**(-k*dp_epsilon)*(est_clean_acc + (dp_delta*c_bound)/(np.e**(dp_epsilon)-1))-(dp_delta*c_bound)/(np.e**(dp_epsilon)-1)
        p2 = 0
        return max(p1, p2)
    
    certified_radius_list = list(range(radius_range))
    certified_acc_list = []
    for k in range(radius_range):
        certified_acc_list.append(dp_baseline_certified_acc(k))
    return certified_acc_list, certified_radius_list


def plot_certified_acc(c_acc_lists, c_rad_lists, name_list, plot_path, xlabel='Number of poisoned training examples', ylabel='Certified Accuracy'):
    print(plot_path)
    for c_acc_list, c_rad_list, name in zip(c_acc_lists, c_rad_lists, name_list):
        logging.info(f'(Rad, Acc):{list(zip(c_rad_list, c_acc_list))}')
        plt.plot(c_rad_list, c_acc_list, label=name)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.savefig(plot_path, bbox_inches='tight')
    plt.clf()


def certify(method_name):
    pred_data = aggregate_result[:, :10]
    pred = np.argmax(pred_data, axis=1)
    gt = aggregate_result[:, 10]
    logging.info(f"Clean acc: {(gt == pred).sum() / len(pred)}")

    certified_poisoning_size_array = np.zeros([num_data], dtype=np.int)
    delta_l, delta_s = (
        1e-50,
        1e-50,
    )  # for simplicity, we use 1e-50 for both delta_l and delta_s, they are actually smaller than 1e-50 in mnist.

    for idx in tqdm(range(num_data)):
        ls = aggregate_result[idx][-1]
        class_freq = aggregate_result[idx][:-1]
        if method_name != 'bagging':
            CI = multi_ci(class_freq, float(args.alpha))
        else:
            CI = multi_ci_bagging(class_freq, float(args.alpha))
        pABar = CI[ls][0]
        probability_bar = CI[:, 1] + delta_s
        probability_bar = np.clip(probability_bar, a_min=-1, a_max=1 - pABar)
        probability_bar[ls] = pABar - delta_l
        probability_bar_copy = np.array(probability_bar, copy=True)
        if method_name == 'dp':
            rd = CertifyRadiusDP(args, ls, probability_bar, dp_epsilon, 1e-5)
        elif method_name == 'dp_baseline_size_one':
            rd = CertifyRadiusDP_baseline(args, ls, probability_bar, dp_epsilon, 1e-5)
        elif method_name == 'rdp':
            rd = CertifyRadiusRDP(args, ls, probability_bar,
                                  rdp_steps, args.sample_rate, args.sigma)
        elif method_name == 'rdp_gp':
            rd = CertifyRadiusRDP_GP(args, ls, probability_bar,
                                  rdp_steps, args.sample_rate, args.sigma)
        elif method_name == 'best':
            rd1 = CertifyRadiusDP(args, ls, probability_bar, dp_epsilon, 1e-5)
            rd2 = CertifyRadiusRDP(
                args, ls, probability_bar_copy, rdp_steps, args.sample_rate, args.sigma)
            rd = max(rd1, rd2)
        elif method_name == 'bagging':
            rd = CertifyRadiusBS(ls, probability_bar, args.sub_training_size, args.training_size)
        elif method_name == 'dp_bagging':
            rd = CertifyRadiusDPBS(args, ls, probability_bar, args.sub_training_size, args.training_size, dp_epsilon, 1e-5, rdp_steps, args.sample_rate, args.sigma)
        else:
            logging.warn(f'Invalid certify method name {method_name}')
            exit(1)
        certified_poisoning_size_array[idx] = rd
        # print('radius:', rd)
        # exit()

    certified_acc_list, certified_radius_list = certified_acc_against_radius(
        certified_poisoning_size_array)

    logging.info(f'Clean acc: {(gt == pred).sum() / len(pred)}')
    logging.info(
        f'{method_name}: certified_poisoning_size_list:\n{certified_radius_list}')
    logging.info(
        f'{method_name}: certified_acc_list_dp:\n{certified_acc_list}')
    return certified_poisoning_size_array


if __name__ == "__main__":
    # main folder
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
    elif args.train_mode == 'Sub-DP-no-amp':
        result_folder = (
            f"{args.results_folder}/{args.model_name}_{args.lr}_{args.sigma}_"
            f"{args.max_per_sample_grad_norm}_{args.sample_rate}_{args.epochs}_{args.sub_training_size}_{args.n_runs}_no_amp"
        )
    else:
        exit('Invalid Method name.')
    print(result_folder)

    # log file for this experiment
    logging.basicConfig(
        filename=f"{result_folder}/certify.log", filemode='w', level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())

    # laod data
    aggregate_result = np.load(f"{result_folder}/aggregate_result.npy")
    num_class = aggregate_result.shape[1] - 1
    num_data = aggregate_result.shape[0]

    if args.train_mode in ['DP', 'Sub-DP', 'Sub-DP-no-amp']:
        dp_epsilon = np.load(f"{result_folder}/dp_epsilon.npy")
        rdp_alphas = np.load(f"{result_folder}/rdp_alphas.npy")
        rdp_epsilons = np.load(f"{result_folder}/rdp_epsilons.npy")
        rdp_steps = np.load(f"{result_folder}/rdp_steps.npy")

        # log params
        logging.info(
            f'lr: {args.lr} sigma: {args.sigma} C: {args.max_per_sample_grad_norm} sample_rate: {args.sample_rate} epochs: {args.epochs} n_runs: {args.n_runs}')
        logging.info(f'dp  epsilon: {dp_epsilon}')
        logging.info(f'rdp epsilons: {rdp_epsilons}')
        logging.info(f'rdp steps: {rdp_steps}')
        logging.info(f'aggregate results:\n{aggregate_result}')
    else:
        logging.info(f'aggregate results:\n{aggregate_result}')

    # Certify
    if not args.plot:
        if args.train_mode in ['DP', 'Sub-DP', 'Sub-DP-no-amp']:
            np.save(f"{result_folder}/dp_cpsa.npy", certify('dp'))
            np.save(f"{result_folder}/rdp_cpsa.npy", certify('rdp'))    
            np.save(f"{result_folder}/rdp_gp_cpsa.npy", certify('rdp_gp'))  
            # np.save(f"{result_folder}/dp_baseline_size_one_cpsa.npy", certify('dp_baseline_size_one'))
            # np.save(f"{result_folder}/best_dp_cpsa.npy", certify('best'))
            if args.train_mode == 'Sub-DP':
                np.save(f"{result_folder}/dp_bagging_cpsa.npy", certify('dp_bagging'))
        elif args.train_mode == 'Bagging':
            np.save(f"{result_folder}/bagging_cpsa.npy", certify('bagging'))

    # Plot
    else:
        if args.train_mode in ['DP', 'Sub-DP', 'Sub-DP-no-amp']:

            method_name = ['DP-Bagging', 'RDP', 'DP', 'Baseline-DP', 'Baseline-RDP-GP']
            acc_list = []
            rad_list = []
            for name in method_name:
                if name == 'DP':
                    acc, rad = certified_acc_against_radius(np.load(f"{result_folder}/dp_cpsa.npy"), radius_range=args.radius_range)
                elif name == 'RDP':
                    acc, rad = certified_acc_against_radius(np.load(f"{result_folder}/rdp_cpsa.npy"), radius_range=args.radius_range)
                elif name == 'Baseline-RDP-GP':
                    acc, rad = certified_acc_against_radius(np.load(f"{result_folder}/rdp_gp_cpsa.npy"), radius_range=args.radius_range)
                elif name == 'Baseline-DP':
                    acc, rad = certified_acc_against_radius_dp_baseline(np.load(f"{result_folder}/acc_list.npy"), dp_epsilon, radius_range=args.radius_range)
                elif name == 'Baseline-DP-size-one':
                    acc, rad = certified_acc_against_radius(np.load(f"{result_folder}/dp_baseline_size_one_cpsa.npy"), radius_range=args.radius_range)
                elif name == 'Baseline-Bagging':
                    acc, rad = certified_acc_against_radius(np.load(f"{result_folder}/bagging_cpsa.npy"), radius_range=args.radius_range)
                elif name == 'Best-DP':
                    acc, rad = certified_acc_against_radius(np.load(f"{result_folder}/best_dp_cpsa.npy"), radius_range=args.radius_range)
                elif name == 'DP-Bagging':
                    acc, rad = certified_acc_against_radius(np.load(f"{result_folder}/dp_bagging_cpsa.npy"), radius_range=args.radius_range)
                else:
                    print('Invalid method name in Plot.')
                acc_list.append(acc)
                rad_list.append(rad)
            plot_certified_acc(acc_list, rad_list, method_name, f"{result_folder}/compare_certified_acc_plot.png")

            # sub_range = [60000, 30000, 20000]
            # cpsa_dp_list = []
            # cpsa_rdp_list = []
            # for sub in sub_range:
            #     cpsa_dp_list.append(np.load(f"{result_folder}/dp_cpsa_{sub}.npy"))
            #     cpsa_rdp_list.append(np.load(f"{result_folder}/rdp_cpsa_{sub}.npy"))
            
            # acc_rad_dp = [certified_acc_against_radius(cpsa_dp, radius_range=args.radius_range) for cpsa_dp in cpsa_dp_list]
            # acc_rad_rdp = [certified_acc_against_radius(cpsa_rdp, radius_range=args.radius_range) for cpsa_rdp in cpsa_rdp_list]

            # plot_certified_acc([x[0] for x in acc_rad_dp], [x[1] for x in acc_rad_dp], [f'Sub-training size {sub}' for sub in sub_range], f"{result_folder}/compare_certified_acc_plot_sub_dp.png")
            # plot_certified_acc([x[0] for x in acc_rad_rdp], [x[1] for x in acc_rad_rdp], [f'Sub-training size {sub}' for sub in sub_range], f"{result_folder}/compare_certified_acc_plot_sub_rdp.png")
            

            
        elif args.train_mode == 'Bagging':
            cpsa_bagging = np.load(f"{result_folder}/bagging_cpsa.npy")
            acc1, rad1 = certified_acc_against_radius(cpsa_bagging, radius_range=args.radius_range)
            plot_certified_acc([acc1], [rad1], ['Bagging'], f"{result_folder}/certified_acc_plot.png")

        # # Optional "epoch V.S. acc" and "epoch V.S. eps" plots
        # epoch_acc_eps = np.load(f"{result_folder}/epoch_acc_eps.npy")
        # acc_list = [x[0] for x in epoch_acc_eps]
        # eps_list = [x[1] for x in epoch_acc_eps]
        # epoch_list = list(range(1, len(epoch_acc_eps)+1))
        # plot_certified_acc([acc_list], [epoch_list], ['acc'], f"{result_folder}/epoch_vs_acc.png", xlabel='Number of epochs', ylabel='Clean Accuracy')
        # plot_certified_acc([eps_list], [epoch_list], ['eps'], f"{result_folder}/epoch_vs_eps.png", xlabel='Number of epochs', ylabel='DP epsilon')

        # # Optional "sub-training-size V.S. acc" plot
        # acc_lists = []
        # subset_lists = []
        # for sigma in [0.5, 1.0, 2.0]:
        #     subset_acc = np.load(f"{result_folder}/subset_acc_list_{sigma}.npy")
        #     subset_list = [x[0] for x in subset_acc]
        #     acc_list = [x[1] for x in subset_acc]
        #     acc_lists.append(acc_list)
        #     subset_lists.append(subset_list)
        # plot_certified_acc(acc_lists, subset_lists, ['Sigma-0.5', 'Sigma-1.0', 'Sigma-2.0'], f"{result_folder}/subset_vs_acc.png", xlabel='Size of sub-training set', ylabel='Clean Accuracy')

        # subset_acc = np.load(f"{result_folder}/subset_acc_list.npy")
        # subset_list = [x[0] for x in subset_acc]
        # acc_list = [x[1] for x in subset_acc]
        # plot_certified_acc([acc_list], [subset_list], ['Sigma-2.0'], f"{result_folder}/subset_vs_acc.png", xlabel='Size of sub-training set', ylabel='Clean Accuracy')