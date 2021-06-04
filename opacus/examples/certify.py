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
parser.add_argument("--alpha", type=str, default="0.001")
parser.add_argument("--exp-name", type=str, default="SampleConvNet_0.1_1.0_1.0_0.001_1_1000")
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
    r, e, d, pl, ps = np.float(radius_value), np.float(epsilon), np.float(delta), np.float(p_l_value), np.float(p_s_value)

    group_eps = e * r
    # group_delta = r * np.e**((r-1)*eps) * delta
    group_delta = d * r

    # print(r, e, d, pl, ps)

    val = pl-group_delta-ps*(np.e**(2*group_eps))-group_delta*(np.e**group_eps)

    if val > 0:
        return True
    else:
        return False

def check_condition_rdp(radius_value, epsilon, alpha, p_l_value, p_s_value):
    k, e, a, pl, ps = np.float(radius_value), np.float(epsilon), np.float(alpha), np.float(p_l_value), np.float(p_s_value)
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
    low, high = 0, 10000
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


def _CertifyRadiusRDP(ls, probability_bar, epsilon, alpha):
    radius = 0
    p_ls = probability_bar[ls]
    probability_bar[ls] = -1
    runner_up_prob = np.amax(probability_bar)
    if p_ls <= runner_up_prob:
        return -1
    for radius in range(10000, 0, -1):
        if check_condition_rdp(radius, epsilon, alpha, p_ls, runner_up_prob):
            return radius
    return 0
    
    # # this is where to calculate the r
    # low, high = 0, 60000
    # while low <= high:
    #     radius = math.ceil((low + high) / 2.0)
    #     if check_condition_rdp(radius, epsilon, alpha, p_ls, runner_up_prob):
    #         low = radius + 0.1
    #     else:
    #         high = radius - 1
    # radius = math.floor(low)
    # if check_condition_rdp(radius, epsilon, alpha, p_ls, runner_up_prob):
    #     return radius
    # else:
    #     print("error")
    #     raise ValueError

def CertifyRadiusRDP(ls, probability_bar, epsilons, alphas):
    rs = []
    for eps, alp in zip(epsilons[500:], alphas[500:]):
        # print(alp)
        rs.append(_CertifyRadiusRDP(ls, probability_bar, eps, alp))
    return max(rs)


if __name__ == "__main__":

    folder_path = f"../results/mnist/{args.exp_name}"
    aggregate_result_path = f"{folder_path}/aggregate_result.npy"
    rdp_alphas_path = f"{folder_path}/alphas.npy"
    rdp_epsilons_path = f"{folder_path}/rdp_epsilons.npy"
    results_file_path = f"{folder_path}/results.txt"

    aggregate_result = np.load(aggregate_result_path)
    rdp_alphas = np.load(rdp_alphas_path)
    rdp_epsilons = np.load(rdp_epsilons_path)

    pred_data = aggregate_result[:, :10]
    pred = np.argmax(pred_data, axis=1)
    gt = aggregate_result[:, 10]

    num_class = aggregate_result.shape[1] - 1
    num_data = aggregate_result.shape[0]

    certified_poisoning_size_array_dp = np.zeros([num_data], dtype=np.int)
    delta_l, delta_s = (
        1e-50,
        1e-50,
    )  # for simplicity, we use 1e-50 for both delta_l and delta_s, they are actually smaller than 1e-50 in mnist.

    un_count = 0

    for idx in range(num_data):
        ls = aggregate_result[idx][-1]
        class_freq = aggregate_result[idx][:-1]
        CI = multi_ci(class_freq, float(args.alpha))
        pABar = CI[ls][0]
        probability_bar = CI[:, 1] + delta_s
        probability_bar = np.clip(probability_bar, a_min=-1, a_max=1 - pABar)
        probability_bar[ls] = pABar - delta_l
        probability_bar_dp = np.array(probability_bar, copy=True)
        rd = CertifyRadiusDP(ls, probability_bar_dp, 0.00677825724599742, 1e-5)    

        certified_poisoning_size_array_dp[idx] = rd
        # print(idx)
        print('dp:', rd)
        # exit()
        

    certified_poisoning_size_list = list(range(50))
    certified_acc_list_dp = []

    for radius in certified_poisoning_size_list:
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
    print('DP:', certified_acc_list_dp)
    with open(results_file_path, 'w') as f:
        f.write('clean acc:{:.5f}\n'.format((gt == pred).sum() / len(pred)))
        f.write('certified_poisoning_size_list:{}\n'.format(certified_poisoning_size_list))
        f.write('certified_acc_list_dp:        {}\n'.format(certified_acc_list_dp))
