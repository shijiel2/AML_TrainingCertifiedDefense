import matplotlib.pyplot as plt
import json
plt.rcParams["figure.figsize"] = (20,10)

results_folder = "./results/stats_result/mnist/"
f_names = ['k_acc_dict.json', 'k_loss_dict.json', 'k_time_dict.json']

for i, f_name in enumerate(f_names):
    f_path = results_folder + f_name
    with open(f_path, 'r') as fout:
        f_dict = json.load(fout)
    for k, v in f_dict.items():
        f_dict[k] = sum(v) / len(v)
    
    plt.figure(i)
    plt.plot(list(f_dict.keys()), list(f_dict.values()))
    plt.xticks(rotation=90)
    plt.tick_params(axis='x', labelsize=8)
    ab = list(zip(list(f_dict.keys()), list(f_dict.values())))
    for i in range(0, len(ab), 3):
        a, b = ab[i]
        plt.text(a, b, "{:.2f}".format(b))

    plt.savefig(f_path.replace('.json', '.png'), dpi=200)

