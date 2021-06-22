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


parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=str, default="0.001")
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
args = parser.parse_args()


def multi_ci(counts, alpha):
    multi_list = []
    n = np.sum(counts)
    l = len(counts)
    for i in range(l):
        multi_list.append(
            proportion_confint(
                # counts[i],
                min(max(counts[i], 1e-10), n - 1e-10),
                n,
                alpha=alpha / 2,
                method="beta",
            )
        )
    return np.array(multi_list)


def check_condition_dp(radius_value, epsilon, delta, p_l_value, p_s_value):
    r, e, d, pl, ps = np.float(radius_value), np.float(epsilon), np.float(
        delta), np.float(p_l_value), np.float(p_s_value)

    group_eps = e * r
    group_delta = d * r

    # print(r, e, d, pl, ps)

    val = pl-ps*(np.e**(2*group_eps))-group_delta*(1+np.e**group_eps)

    if val > 0:
        return True
    else:
        return False


def check_condition_rdp(radius_value, epsilon, alpha, p_l_value, p_s_value):
    k, e, a, pl, ps = np.float(radius_value), np.float(epsilon), np.float(
        alpha), np.float(p_l_value), np.float(p_s_value)
    if k == 0:
        return True

    c = np.log2(k)
    if a < 2*k:
        return False

    ga = a / k
    ge = e * 3**c

    val = (np.e**(-ge)) * (pl**(ga/(ga-1))) - (np.e**(ge)) * (ps**((ga-1)/ga))

    if val > 0:
        return True
    else:
        return False


def CertifyRadiusDP(ls, probability_bar, epsilon, delta):
    radius = 0
    p_ls = probability_bar[ls]
    probability_bar[ls] = -1
    runner_up_prob = np.amax(probability_bar)
    if p_ls <= runner_up_prob:
        return -1
    # this is where to calculate the r
    low, high = 0, 1000
    while low <= high:
        radius = math.ceil((low + high) / 2.0)
        if check_condition_dp(radius, epsilon, delta, p_ls, runner_up_prob):
            low = radius + 0.1
        else:
            high = radius - 1
    radius = math.floor(low)
    if check_condition_dp(radius, epsilon, delta, p_ls, runner_up_prob):
        return radius
    else:
        print("error")
        raise ValueError


def CertifyRadiusRDP(ls, probability_bar, steps, sample_rate, sigma):
    p1 = probability_bar[ls]
    probability_bar[ls] = -1
    p2 = np.amax(probability_bar)
    if p1 <= p2:
        return -1

    valid_radius = set()
    for alpha in [1 + x/2 for x in range(1, 20)]:
        for delta in [x / 100.0 for x in range(1, 10)]:
        # for delta in [0]:
            # binary search for radius
            low, high = 0, 1000
            while low <= high:
                radius = math.ceil((low + high) / 2.0)
                if PrivacyEngine.is_rdp_certified_radius(radius=radius, sample_rate=sample_rate, steps=steps, alpha=alpha, delta=delta, sigma=sigma, p1=p1, p2=p2):
                    low = radius + 0.1
                else:
                    high = radius - 1
            radius = math.floor(low)
            if PrivacyEngine.is_rdp_certified_radius(radius=radius, sample_rate=sample_rate, steps=steps, alpha=alpha, delta=delta, sigma=sigma, p1=p1, p2=p2):
                valid_radius.add((radius, alpha, delta))
            elif radius == 0:
                valid_radius.add((radius, alpha, delta))
            else:
                print("error", (radius, alpha, delta))
                raise ValueError

    if len(valid_radius) > 0:
        max_radius = max(valid_radius, key=lambda x: x[0])[0]
        # for x in valid_radius:
        #     if x[0] == max_radius:
        #         print(x)
        if max_radius == 0:
            return -1
        else:
            return max_radius
    else:
        return -1


def certified_acc_against_radius(certified_poisoning_size_array, radius_range=100):
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


def plot_certified_acc(cpsa_list, plot_path):
    for cpsa in cpsa_list:
        c_acc_list, c_rad_list = certified_acc_against_radius(cpsa)
        plt.plot(c_rad_list, c_acc_list)
    plt.savefig(plot_path, bbox_inches='tight')
    plt.clf()


def certify(certify_method, method_name):
    pred_data = aggregate_result[:, :10]
    pred = np.argmax(pred_data, axis=1)
    gt = aggregate_result[:, 10]

    certified_poisoning_size_array = np.zeros([num_data], dtype=np.int)
    delta_l, delta_s = (
        1e-50,
        1e-50,
    )  # for simplicity, we use 1e-50 for both delta_l and delta_s, they are actually smaller than 1e-50 in mnist.

    for idx in tqdm(range(num_data)):
        ls = aggregate_result[idx][-1]
        class_freq = aggregate_result[idx][:-1]
        CI = multi_ci(class_freq, float(args.alpha))
        pABar = CI[ls][0]
        probability_bar = CI[:, 1] + delta_s
        probability_bar = np.clip(probability_bar, a_min=-1, a_max=1 - pABar)
        probability_bar[ls] = pABar - delta_l
        probability_bar_dp = np.array(probability_bar, copy=True)
        if method_name == 'DP':
            rd = certify_method(ls, probability_bar_dp, dp_epsilon, 1e-5)
        elif method_name == 'RDP':
            rd = certify_method(ls, probability_bar_dp,
                                rdp_steps, args.sample_rate, args.sigma)
        else:
            logging.warn(f'Invalid certify method name {method_name}')
            exit(1)
        certified_poisoning_size_array[idx] = rd
        # print('radius:', rd)
        # exit()

    certified_acc_list, certified_radius_list = certified_acc_against_radius(certified_poisoning_size_array)

    logging.info(f'Clean acc: {(gt == pred).sum() / len(pred)}')
    logging.info(
        f'{method_name}: certified_poisoning_size_list:\n{certified_radius_list}')
    logging.info(
        f'{method_name}: certified_acc_list_dp:\n{certified_acc_list}')
    return certified_poisoning_size_array

if __name__ == "__main__":
    # main folder
    result_folder = (
        f"{args.results_folder}/{args.model_name}_{args.lr}_{args.sigma}_"
        f"{args.max_per_sample_grad_norm}_{args.sample_rate}_{args.epochs}_{args.n_runs}"
    )
    print(result_folder)
    # log file for this experiment
    logging.basicConfig(
        filename=f"{result_folder}/certify.log", filemode='w', level=logging.INFO)
    # laod data
    aggregate_result = np.load(f"{result_folder}/aggregate_result.npy")
    dp_epsilon = np.load(f"{result_folder}/dp_epsilon.npy")
    rdp_alphas = np.load(f"{result_folder}/rdp_alphas.npy")
    rdp_epsilons = np.load(f"{result_folder}/rdp_epsilons.npy")
    rdp_steps = np.load(f"{result_folder}/rdp_steps.npy")
    
    num_class = aggregate_result.shape[1] - 1
    num_data = aggregate_result.shape[0]
    
    # log params
    logging.info(
        f'lr: {args.lr} sigma: {args.sigma} C: {args.max_per_sample_grad_norm} sample_rate: {args.sample_rate} epochs: {args.epochs} n_runs: {args.n_runs}')
    logging.info(f'dp  epsilon: {dp_epsilon}')
    logging.info(f'rdp alphas: {rdp_alphas}')
    logging.info(f'rdp epsilons: {rdp_epsilons}')
    logging.info(f'rdp steps: {rdp_steps}')
    logging.info(f'aggregate results:\n{aggregate_result}')

    if args.plot:
        cpsa_dp = np.load(f"{result_folder}/dp_cpsa.npy")
        cpsa_rdp = np.load(f"{result_folder}/rdp_cpsa.npy")
        plot_certified_acc([cpsa_dp, cpsa_rdp], f"{result_folder}/certified_acc_plot.png")
    
    else:
        # np.save(f"{result_folder}/dp_cpsa.npy", certify(CertifyRadiusDP, 'DP'))
        np.save(f"{result_folder}/rdp_cpsa.npy", certify(CertifyRadiusRDP, 'RDP'))



