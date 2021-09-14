from __future__ import print_function
from os import EX_OSFILE
import numpy as np
from numpy.core.numeric import binary_repr
from statsmodels.stats.proportion import (
    proportion_confint
)
import math

from opacus import PrivacyEngine
import scipy.stats

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

def dp_amplify(epsilon, delta, m, n):
    
    mu = m / n
    delta_new = mu * delta
    epsilon_new = np.log(1 + mu * (np.e**epsilon - 1))

    return epsilon_new, delta_new

def rdp_amplify(alpha, m, n, sample_rate, sigma):
    
    prob = m / n

    # print(f'm:{m}, n:{n}, prob:{prob}')

    from autodp import utils

    def func(alpha):
        rdp = PrivacyEngine._get_renyi_divergence(sample_rate=sample_rate, noise_multiplier=sigma, alphas=[alpha])
        eps = rdp.cpu().detach().numpy()[0]
        return eps

    def cgf(x):
        return x * func(x+1)

    def subsample_epsdelta(eps,delta,prob):
        if prob == 0:
            return 0,0
        return np.log(1+prob*(np.exp(eps)-1)), prob*delta

    def subsample_func_int(x):
        # output the cgf of the subsampled mechanism
        mm = int(x)
        eps_inf = func(np.inf)

        moments_two = 2 * np.log(prob) + utils.logcomb(mm,2) \
                        + np.minimum(np.log(4) + func(2.0) + np.log(1-np.exp(-func(2.0))),
                                    func(2.0) + np.minimum(np.log(2),
                                                2 * (eps_inf+np.log(1-np.exp(-eps_inf)))))
        moment_bound = lambda j: np.minimum(j * (eps_inf + np.log(1-np.exp(-eps_inf))),
                                            np.log(2)) + cgf(j - 1) \
                                    + j * np.log(prob) + utils.logcomb(mm, j)
        moments = [moment_bound(j) for j in range(3, mm + 1, 1)]
        return np.minimum((x-1)*func(x), utils.stable_logsumexp([0,moments_two] + moments))
    
    def subsample_func(x):
        # This function returns the RDP at alpha = x
        # RDP with the linear interpolation upper bound of the CGF

        epsinf, tmp = subsample_epsdelta(func(np.inf),0,prob)

        if np.isinf(x):
            return epsinf
        if prob == 1.0:
            return func(x)

        if (x >= 1.0) and (x <= 2.0):
            return np.minimum(epsinf, subsample_func_int(2.0) / (2.0-1))
        if np.equal(np.mod(x, 1), 0):
            return np.minimum(epsinf, subsample_func_int(x) / (x-1) )
        xc = math.ceil(x)
        xf = math.floor(x)
        return np.min(
            [epsinf,func(x),
                ((x-xf)*subsample_func_int(xc) + (1-(x-xf))*subsample_func_int(xf)) / (x-1)]
        )

    return alpha, subsample_func(alpha)



def check_condition_dp(args, radius_value, epsilon, delta, p_l_value, p_s_value):
    if radius_value == 0:
        return True

    r, e, d, pl, ps = np.float(radius_value), np.float(epsilon), np.float(
        delta), np.float(p_l_value), np.float(p_s_value)
    
    if args.train_mode == 'Sub-DP':
        e, d = dp_amplify(e, d, args.sub_training_size, args.training_size)

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


def check_condition_dp_baseline(args, radius_value, epsilon, delta, p_l_value, p_s_value):
    if radius_value == 0:
        return True

    r, e, d, pl, ps = np.float(radius_value), np.float(epsilon), np.float(
        delta), np.float(p_l_value), np.float(p_s_value)

    lower = np.e**(-r*e) * (pl + d / (np.e**e - 1))
    upper = np.e**(r*e) * (ps + d / (np.e**e - 1)) 

    # print(r, e, d, pl, ps)
    try:
        val = lower - upper
    except Exception:
        val = 0

    if val > 0:
        return True
    else:
        return False


def check_condition_rdp(args, radius, sample_rate, steps, alpha, delta, sigma, p1, p2):

    if radius == 0:
        return True
    
    sample_rate = 1 - (1 - sample_rate)**radius

    if args.train_mode == 'DP' or args.train_mode == 'Sub-DP-no-amp':
        rdp = PrivacyEngine._get_renyi_divergence(
            sample_rate=sample_rate, noise_multiplier=sigma, alphas=[alpha]) * steps
        eps = rdp.cpu().detach().numpy()[0]
    elif args.train_mode == 'Sub-DP':
        _, eps = rdp_amplify(alpha, args.sub_training_size, args.training_size, sample_rate, sigma)
        eps *= steps

    val = np.e**(-eps) * p1**(alpha/(alpha-1)) - (np.e**eps * p2)**((alpha-1)/alpha)
    if val >= 0:
        return True
    else:
        return False


def check_condition_rdp_gp(args, radius, sample_rate, steps, alpha, delta, sigma, p1, p2):

    if radius == 0:
        return True

    if args.train_mode == 'DP' or args.train_mode == 'Sub-DP-no-amp':
        rdp = PrivacyEngine._get_renyi_divergence(
            sample_rate=sample_rate, noise_multiplier=sigma, alphas=[alpha]) * steps
        eps = rdp.cpu().detach().numpy()[0]
    elif args.train_mode == 'Sub-DP':
        _, eps = rdp_amplify(alpha, args.sub_training_size, args.training_size, sample_rate, sigma)
        eps *= steps

    alpha = alpha / radius
    eps = 3**(np.log2(radius)) * eps

    if alpha <= 1:
        return False

    val = np.e**(-eps) * p1**(alpha/(alpha-1)) - (np.e**eps * p2)**((alpha-1)/alpha)
    if val >= 0:
        return True
    else:
        return False


def check_condition_dp_bagging(radius_value, k_value, n_value, p_l_value, p_s_value, dp_rad):

    if radius_value == 0:
        return True
    
    import math
    def nCr(n,r):
        f = math.factorial
        return int(f(n) / f(r) / f(n-r))

    def binoSum(k, x_start, p):
        prob_sum = 0
        for x in range(x_start, k+1):
            prob = nCr(k, x) * p**x * (1-p)**(k-x)
            prob_sum += prob
        return prob_sum

    p3 = binoSum(k_value, (k_value-dp_rad), (n_value-radius_value)/n_value)
    lower = p_l_value - (1-p3)
    upper = p_s_value + (1-p3)
    
    try:
        val = lower - upper
    except Exception:
        val = 0

    if val > 0:
        return True
    else:
        return False


def check_condition_bagging(radius_value, k_value, n_value, p_l_value, p_s_value):

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

def CertifyRadiusDP(args, ls, probability_bar, epsilon, delta):
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
        if check_condition_dp(args, radius, epsilon, delta, p_ls, runner_up_prob):
            low = radius + 0.1
        else:
            high = radius - 1
    radius = math.floor(low)
    if check_condition_dp(args, radius, epsilon, delta, p_ls, runner_up_prob):
        return radius
    else:
        print("error")
        raise ValueError


def CertifyRadiusDP_baseline(args, ls, probability_bar, epsilon, delta):
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
        if check_condition_dp_baseline(args, radius, epsilon, delta, p_ls, runner_up_prob):
            low = radius + 0.1
        else:
            high = radius - 1
    radius = math.floor(low)
    if check_condition_dp_baseline(args, radius, epsilon, delta, p_ls, runner_up_prob):
        return radius
    else:
        print("error")
        raise ValueError


def CertifyRadiusRDP(args, ls, probability_bar, steps, sample_rate, sigma):
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
                if check_condition_rdp(args, radius=radius, sample_rate=sample_rate, steps=steps, alpha=alpha, delta=delta, sigma=sigma, p1=p1, p2=p2):
                    low = radius + 0.1
                else:
                    high = radius - 1
            radius = math.floor(low)
            if check_condition_rdp(args, radius=radius, sample_rate=sample_rate, steps=steps, alpha=alpha, delta=delta, sigma=sigma, p1=p1, p2=p2):
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


def CertifyRadiusRDP_GP(args, ls, probability_bar, steps, sample_rate, sigma):
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
                if check_condition_rdp_gp(args, radius=radius, sample_rate=sample_rate, steps=steps, alpha=alpha, delta=delta, sigma=sigma, p1=p1, p2=p2):
                    low = radius + 0.1
                else:
                    high = radius - 1
            radius = math.floor(low)
            if check_condition_rdp_gp(args, radius=radius, sample_rate=sample_rate, steps=steps, alpha=alpha, delta=delta, sigma=sigma, p1=p1, p2=p2):
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


def CertifyRadiusDPBS(args, ls, probability_bar, k, n, epsilon, delta, steps, sample_rate, sigma):
    # first using CertifyRadius_DP to find out the robustness we have in a sub-dataset
    # change train_mode to 'DP' to avoid dp amplification
    args.train_mode = 'DP'
    dp_rad = max(0, CertifyRadiusRDP(args, ls, np.array(probability_bar, copy=True), steps, sample_rate, sigma), CertifyRadiusDP(args, ls, np.array(probability_bar, copy=True), epsilon, delta))
    args.train_mode = 'Sub-DP'

    # DP bagging part
    radius = 0
    p_ls = probability_bar[ls]
    probability_bar[ls] = -1
    runner_up_prob = np.amax(probability_bar)
    if p_ls <= runner_up_prob:
        return -1
    low, high = 0, 1500
    while low <= high:
        radius = math.ceil((low+high)/2.0)
        if check_condition_dp_bagging(radius, k, n, p_ls, runner_up_prob, dp_rad):
            low = radius + 0.1
        else:
            high = radius - 1
    radius = math.floor(low)
    if check_condition_dp_bagging(radius, k, n, p_ls, runner_up_prob, dp_rad):
        return radius
    else:
        print("error")
        raise ValueError
