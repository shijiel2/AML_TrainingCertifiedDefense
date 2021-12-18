from sympy import * 
import numpy as np
import pickle
from tqdm import tqdm


t, mu, sigma = symbols('t, mu, sigma')

# normal_mgf = np.e**(t*mu + 1/2 * sigma**2 * t**2)
# n = 102
# diff_list = [normal_mgf]
# for i in tqdm(range(n)):
#     diff_list.append(diff(diff_list[-1], t))
# pickle.dump(diff_list, open('opacus/results/mgf_diff_list.p', 'wb'))


diff_list = pickle.load(open('opacus/results/mgf_diff_list.p', 'rb'))
print(diff_list[4].evalf(subs={t: 0, mu: 19, sigma: 1}))

