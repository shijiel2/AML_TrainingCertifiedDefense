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
    "--bagging-size",
    type=int,
    default=0,
    help="Size of bagging",
)
parser.add_argument(
    "--disable-dp",
    action="store_true",
    default=False,
    help="Disable privacy training and just train with vanilla SGD",
)
parser.add_argument(
    "--training-size",
    type=int,
    default=60000,
    help="Size of training set",
)
parser.add_argument(
    "--method-name",
    type=str,
    default="DP",
    help="Name of the methods: DP, DP-Baseline, Bagging",
)
parser.add_argument(
    "--radius-range",
    type=int,
    default=20,
    help="Size of training set",
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


def single_ci(counts, alpha):
    a = 1.0 * np.array(counts)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + (1 - alpha)) / 2., n-1)
    return m, m-h, m+h


def multi_ci_bagging(counts, alpha):
    multi_list = []
    n = np.sum(counts)
    l = len(counts)
    for i in range(l):
        multi_list.append(proportion_confint(
            min(max(counts[i], 1e-10), n-1e-10), n, alpha=alpha*2./l, method="beta"))
    return np.array(multi_list)


def check_condition_dp(radius_value, epsilon, delta, p_l_value, p_s_value):
    r, e, d, pl, ps = np.float(radius_value), np.float(epsilon), np.float(
        delta), np.float(p_l_value), np.float(p_s_value)

    group_eps = e * r
    group_delta = d * r

    # print(r, e, d, pl, ps)
    try:
        val = pl-ps*(np.e**(2*group_eps))-group_delta*(1+np.e**group_eps)
    except Exception:
        val = 0

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

def check_condition_bagging(radius_value, k_value, n_value, p_l_value,p_s_value):

    threshold_point = radius_value / (1.0 - np.power(0.5, 1.0/(k_value-1.0)))

    if threshold_point <= n_value: 
        nprime_value = int(n_value)
        value_check = compute_compare_value_bagging(radius_value,nprime_value,k_value,n_value,p_l_value,p_s_value)
    elif threshold_point >=n_value+radius_value:
        nprime_value = int(n_value+radius_value)
        value_check = compute_compare_value_bagging(radius_value,nprime_value,k_value,n_value,p_l_value,p_s_value) 
    else:
        nprime_value_1 = np.ceil(threshold_point)
        value_check_1 = compute_compare_value_bagging(radius_value,nprime_value_1,k_value,n_value,p_l_value,p_s_value)
        nprime_value_2 = np.floor(threshold_point)
        value_check_2 = compute_compare_value_bagging(radius_value,nprime_value_2,k_value,n_value,p_l_value,p_s_value)   
        value_check = max(value_check_1,value_check_2)            
    if value_check<0:
        return True 
    else:
        return False 

def compute_compare_value_bagging(radius_cmp,nprime_cmp,k_cmp,n_cmp,p_l_cmp,p_s_cmp):
    return np.power(float(nprime_cmp)/float(n_cmp),k_cmp) - 2*np.power((float(nprime_cmp)-float(radius_cmp))/float(n_cmp),k_cmp) + 1 - p_l_cmp + p_s_cmp

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
    for alpha in [1 + x for x in range(1, 100)]:
        # for delta in [x / 100.0 for x in range(1, 10)]:
        for delta in [0]:
            # binary search for radius
            low, high = 0, 1000
            while low <= high:
                radius = math.ceil((low + high) / 2.0)
                if PrivacyEngine.is_rdp_certified_radius_2(radius=radius, sample_rate=sample_rate, steps=steps, alpha=alpha, delta=delta, sigma=sigma, p1=p1, p2=p2):
                    low = radius + 0.1
                else:
                    high = radius - 1
            radius = math.floor(low)
            if PrivacyEngine.is_rdp_certified_radius_2(radius=radius, sample_rate=sample_rate, steps=steps, alpha=alpha, delta=delta, sigma=sigma, p1=p1, p2=p2):
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
        return max_radius
    else:
        return 0


def CertifyRadiusBS(ls, probability_bar, k, n):
    radius = 0
    p_ls = probability_bar[ls]
    probability_bar[ls] = -1
    runner_up_prob = np.amax(probability_bar)
    if p_ls <= runner_up_prob:
        return -1
    low, high = 0, 1500
    while low <= high:
        radius = math.ceil((low+high)/2.0)
        if check_condition_bagging(radius, k, n, p_ls, runner_up_prob):
            low = radius + 0.1
        else:
            high = radius - 1
    radius = math.floor(low)
    if check_condition_bagging(radius, k, n, p_ls, runner_up_prob):
        return radius
    else:
        print("error")
        raise ValueError


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
    plt.legend(loc="upper left")
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
            rd = CertifyRadiusDP(ls, probability_bar, dp_epsilon, 1e-5)
        elif method_name == 'rdp':
            rd = CertifyRadiusRDP(ls, probability_bar,
                                  rdp_steps, args.sample_rate, args.sigma)
        elif method_name == 'best':
            rd1 = CertifyRadiusDP(ls, probability_bar, dp_epsilon, 1e-5)
            rd2 = CertifyRadiusRDP(
                ls, probability_bar_copy, rdp_steps, args.sample_rate, args.sigma)
            rd = max(rd1, rd2)
        elif method_name == 'bagging':
            rd = CertifyRadiusBS(ls, probability_bar, args.bagging_size, args.training_size)
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
    if args.method_name in ['DP', 'DP-Baseline', 'Epoch_acc_eps', 'Subset_acc']:
        result_folder = (
            f"{args.results_folder}/{args.model_name}_{args.lr}_{args.sigma}_"
            f"{args.max_per_sample_grad_norm}_{args.sample_rate}_{args.epochs}_{args.n_runs}"
        )
    elif args.method_name in ['Bagging']:
        result_folder = (
            f"{args.results_folder}/Bagging_{args.model_name}_{args.lr}_{args.bagging_size}_"
            f"{args.epochs}_{args.n_runs}"
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

    if not args.disable_dp:
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
        if args.method_name == 'DP':
            np.save(f"{result_folder}/dp_cpsa.npy", certify('dp'))
            np.save(f"{result_folder}/rdp_cpsa.npy", certify('rdp'))
            # np.save(f"{result_folder}/best_dp_cpsa.npy", certify('best'))        
        elif args.method_name == 'DP-Baseline':
            pass
        elif args.method_name == 'Bagging':
            np.save(f"{result_folder}/bagging_cpsa.npy", certify('bagging'))

    # Plot
    else:
        if args.method_name == 'DP':
            cpsa_dp = np.load(f"{result_folder}/dp_cpsa.npy")
            cpsa_rdp = np.load(f"{result_folder}/rdp_cpsa.npy")
            # cpsa_best_dp = np.load(f"{result_folder}/best_dp_cpsa.npy")
            # cpsa_bagging = np.load(f"{result_folder}/bagging_cpsa.npy")
            clean_acc_list = np.load(f"{result_folder}/acc_list.npy")
    
            acc1, rad1 = certified_acc_against_radius(cpsa_rdp, radius_range=args.radius_range)
            acc2, rad2 = certified_acc_against_radius(cpsa_dp, radius_range=args.radius_range)
            acc3, rad3 = certified_acc_against_radius_dp_baseline(clean_acc_list, dp_epsilon, radius_range=args.radius_range)
            plot_certified_acc([acc1, acc2, acc3], [rad1, rad2, rad3], ['RDP', 'DP', 'Baseline-DP'], f"{result_folder}/compare_certified_acc_plot.png")
            
        elif args.method_name == 'DP-Baseline':
            clean_acc_list = np.load(f"{result_folder}/acc_list.npy")
            acc1, rad1 = certified_acc_against_radius_dp_baseline(clean_acc_list, dp_epsilon, radius_range=args.radius_range)
            plot_certified_acc([acc1], [rad1], ['DP-Baseline'], f"{result_folder}/dp_baseline_certified_acc_plot.png")
            
        elif args.method_name == 'Bagging':
            cpsa_bagging = np.load(f"{result_folder}/bagging_cpsa.npy")
            acc1, rad1 = certified_acc_against_radius(cpsa_bagging, radius_range=args.radius_range)
            plot_certified_acc([acc1], [rad1], ['Bagging'], f"{result_folder}/certified_acc_plot.png")

        elif args.method_name == 'Epoch_acc_eps':
            epoch_acc_eps = np.load(f"{result_folder}/epoch_acc_eps.npy")
            acc_list = [x[0] for x in epoch_acc_eps]
            eps_list = [x[1] for x in epoch_acc_eps]
            epoch_list = list(range(1, len(epoch_acc_eps)+1))
            plot_certified_acc([acc_list], [epoch_list], ['acc'], f"{result_folder}/epoch_vs_acc.png", xlabel='Number of epochs', ylabel='Clean Accuracy')
            plot_certified_acc([eps_list], [epoch_list], ['eps'], f"{result_folder}/epoch_vs_eps.png", xlabel='Number of epochs', ylabel='DP epsilon')

        elif args.method_name == 'Subset_acc':
            acc_lists = []
            subset_lists = []
            for epoch in [20, 50, 100]:
                subset_acc = np.load(f"{result_folder}/subset_acc_list_{epoch}.npy")
                subset_list = [x[0] for x in subset_acc]
                acc_list = [x[1] for x in subset_acc]
                acc_lists.append(acc_list)
                subset_lists.append(subset_list)
                
            plot_certified_acc(acc_lists, subset_lists, ['Epochs-20', 'Epochs-50', 'Epochs-100'], f"{result_folder}/subset_vs_acc.png", xlabel='Size of sub-training set', ylabel='Clean Accuracy')

