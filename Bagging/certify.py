from __future__ import print_function
import numpy as np
from statsmodels.stats.proportion import (
    proportion_confint,
    multinomial_proportions_confint,
)
import argparse
import math
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="mnist")
parser.add_argument("--k", type=str, default="30")
parser.add_argument("--poison_size", default="0")
parser.add_argument("--n", type=str, default="60000")
parser.add_argument("--ns", type=str, default="1000")
parser.add_argument("--alpha", type=str, default="0.001")

args = parser.parse_args()


def multi_ci(counts, alpha):
    multi_list = []
    n = np.sum(counts)
    l = len(counts)
    for i in range(l):
        multi_list.append(
            proportion_confint(
                min(max(counts[i], 1e-10), n - 1e-10),
                n,
                alpha=alpha * 2.0 / l,
                method="beta",
            )
        )
    return np.array(multi_list)


def Compute_compare_value(radius_cmp, nprime_cmp, k_cmp, n_cmp, p_l_cmp, p_s_cmp):
    return (
        np.power(float(nprime_cmp) / float(n_cmp), k_cmp)
        - 2 * np.power((float(nprime_cmp) - float(radius_cmp)) /
                       float(n_cmp), k_cmp)
        + 1
        - p_l_cmp
        + p_s_cmp
    )


def Check_condition(radius_value, k_value, n_value, p_l_value, p_s_value):

    threshold_point = radius_value / \
        (1.0 - np.power(0.5, 1.0 / (k_value - 1.0)))

    if threshold_point <= n_value:
        nprime_value = int(n_value)
        value_check = Compute_compare_value(
            radius_value, nprime_value, k_value, n_value, p_l_value, p_s_value
        )
    elif threshold_point >= n_value + radius_value:
        nprime_value = int(n_value + radius_value)
        value_check = Compute_compare_value(
            radius_value, nprime_value, k_value, n_value, p_l_value, p_s_value
        )
    else:
        nprime_value_1 = np.ceil(threshold_point)
        value_check_1 = Compute_compare_value(
            radius_value, nprime_value_1, k_value, n_value, p_l_value, p_s_value
        )
        nprime_value_2 = np.floor(threshold_point)
        value_check_2 = Compute_compare_value(
            radius_value, nprime_value_2, k_value, n_value, p_l_value, p_s_value
        )
        value_check = max(value_check_1, value_check_2)
    if value_check < 0:
        return True
    else:
        return False

def check_condition_dp(radius_value, k_value, n_value, p_l_value, p_s_value):
    r, k, n, pl, ps = np.float(radius_value), np.float(k_value), np.float(n_value), np.float(p_l_value), np.float(p_s_value)
    eps = k*np.log((n+1)/n)
    delta = 1 - ((n-1)/n)**k

    # print(eps)
    # print(delta)

    group_eps = eps * r
    # group_delta = r * np.e**((r-1)*eps) * delta
    group_delta = delta * r

    val = pl-group_delta-ps*(np.e**(2*group_eps))-group_delta*(np.e**group_eps)

    if val > 0:
        return True
    else:
        return False

def CertifyRadiusDP(ls, probability_bar, k, n):
    radius = 0
    p_ls = probability_bar[ls]
    probability_bar[ls] = -1
    runner_up_prob = np.amax(probability_bar)
    if p_ls <= runner_up_prob:
        return -1
    # this is where to calculate the r
    low, high = 0, 10000
    while low <= high:
        radius = math.ceil((low + high) / 2.0)
        if check_condition_dp(radius, k, n, p_ls, runner_up_prob):
            low = radius + 0.1
        else:
            high = radius - 1
    radius = math.floor(low)
    if check_condition_dp(radius, k, n, p_ls, runner_up_prob):
        return radius
    else:
        print("error")
        raise ValueError

def CertifyRadiusBS(ls, probability_bar, k, n):
    radius = 0
    p_ls = probability_bar[ls]
    probability_bar[ls] = -1
    runner_up_prob = np.amax(probability_bar)
    if p_ls <= runner_up_prob:
        return -1
    # this is where to calculate the r
    low, high = 0, 1500
    while low <= high:
        radius = math.ceil((low + high) / 2.0)
        if Check_condition(radius, k, n, p_ls, runner_up_prob):
            low = radius + 0.1
        else:
            high = radius - 1
    radius = math.floor(low)
    if Check_condition(radius, k, n, p_ls, runner_up_prob):
        return radius
    else:
        print("error")
        raise ValueError


if __name__ == "__main__":

    folder_path = "./results/aggregate_result/{}/k_{}_poison-size_{}".format(args.dataset, args.k, args.poison_size)
    input_file = "{}/aggregate_batch_k_{}_start_0_end_{}.npz".format(folder_path, args.k, args.ns)
    dstnpz_file = "{}/certified_poisoning_size_k_{}_ns_{}_alpha_{}.npz".format(folder_path, args.k, args.ns, args.alpha)
    results_file = '{}/results.txt'.format(folder_path)

    data = np.load(input_file)["x"]

    pred_data = data[:, :10]
    pred = np.argmax(pred_data, axis=1)
    gt = data[:, 10]

    num_class = data.shape[1] - 1
    num_data = data.shape[0]

    certified_poisoning_size_array = np.zeros([num_data], dtype=np.int)
    certified_poisoning_size_array_dp = np.zeros([num_data], dtype=np.int)
    delta_l, delta_s = (
        1e-50,
        1e-50,
    )  # for simplicity, we use 1e-50 for both delta_l and delta_s, they are actually smaller than 1e-50 in mnist.
    for idx in range(num_data):
        ls = data[idx][-1]
        class_freq = data[idx][:-1]
        CI = multi_ci(class_freq, float(args.alpha) / data.shape[0])
        pABar = CI[ls][0]
        probability_bar = CI[:, 1] + delta_s
        probability_bar = np.clip(probability_bar, a_min=-1, a_max=1 - pABar)
        probability_bar[ls] = pABar - delta_l
        probability_bar_dp = np.array(probability_bar, copy=True)
        rb = CertifyRadiusBS(ls, probability_bar, int(args.k), int(args.n))
        rd = CertifyRadiusDP(ls, probability_bar_dp, int(args.k), int(args.n))     
        certified_poisoning_size_array[idx] = rb
        certified_poisoning_size_array_dp[idx] = rd
        # print(idx)
        # print('bg, dp:', rb, rd)
    # exit()
        

    certified_poisoning_size_list = [
        0,
        100,
        200,
        300,
        400,
        500,
        600,
        700,
        800,
        900,
        1000,
        1100,
    ]
    certified_acc_list = []
    certified_acc_list_dp = []

    for radius in certified_poisoning_size_list:
        certified_acc_list.append(
            len(
                certified_poisoning_size_array[
                    np.where(certified_poisoning_size_array >= radius)
                ]
            )
            / float(num_data)
        )
        certified_acc_list_dp.append(
            len(
                certified_poisoning_size_array_dp[
                    np.where(certified_poisoning_size_array_dp >= radius)
                ]
            )
            / float(num_data)
        )

    print("Clean acc:", (gt == pred).sum() / len(pred))
    print(certified_poisoning_size_list)
    print('BG:', certified_acc_list)
    print('DP:', certified_acc_list_dp)
    with open(results_file, 'w') as f:
        f.write('clean acc:{:.5f}\n'.format((gt == pred).sum() / len(pred)))
        f.write('certified_poisoning_size_list:{}\n'.format(certified_poisoning_size_list))
        f.write('certified_acc_list_bagging:   {}\n'.format(certified_acc_list))
        f.write('certified_acc_list_dp:        {}\n'.format(certified_acc_list_dp))
    np.savez(dstnpz_file, x=certified_poisoning_size_array, y=certified_poisoning_size_array_dp)
