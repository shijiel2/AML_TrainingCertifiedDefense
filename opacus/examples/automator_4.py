import itertools
import socket
import sys

# sys.path.append("../..")
import subprocess
from certify_utilis import get_dir, extract_summary, merge_models
# from notification import NOTIFIER
from datetime import datetime
from pathlib import Path


MODE = ['train', 'neval', 'ncertify', 'nplot', 'nablation', 'nsub-acc-test', 'nsummary', 'nmerge']
DATASET = 'cifar10'
TRAIN_MODE = 'Bagging' # DP, Sub-DP, Bagging, Sub-DP-no-amp

# No saving
TRAIN_COMMAND = 'python {dataset}.py --n-runs {n_runs} --epochs {epochs} --sigma {sigma} --sample-rate {sample_rate} --lr {lr} -c {c} --model-name {model_name} --sub-training-size {sub_training_size} --train-mode {train_mode} --results-folder {results_folder} --save-model'

EVAL_COMMAND = 'python {dataset}.py --n-runs {n_runs} --epochs {epochs} --sigma {sigma} --sample-rate {sample_rate} --lr {lr} -c {c} --model-name {model_name} --sub-training-size {sub_training_size} --train-mode {train_mode} --results-folder {results_folder} --load-model'

CERTIFY_COMMAND = 'python certify.py --n-runs {n_runs} --epochs {epochs} --sigma {sigma} --sample-rate {sample_rate} --lr {lr} -c {c} --model-name {model_name} --results-folder {results_folder} --training-size {training_size} --sub-training-size {sub_training_size} --train-mode {train_mode} --mode {mode}'

TRAIN_SUBSET_ACC = 'python {dataset}.py --n-runs {n_runs} --epochs {epochs} --sigma {sigma} --sample-rate {sample_rate} --lr {lr} -c {c} --model-name {model_name} --sub-training-size {sub_training_size} --save-model --train-mode {train_mode} --results-folder {results_folder} --sub-acc-test'

if DATASET == 'mnist':
    results_folder = '../results/mnist'
    model_name = 'LeNet'
    training_size = 60000
    n_runss = [1000]
    epochss = [1]
    sigmas = [2.0] # (sigma, clip): (1.0, 2.4), (2.0, 1.3), (3.0, 0.8), (4.0, 0.7)
    sample_rates = [0.001]
    lrs = [0.1]
    clips = [1.0]
    sub_training_sizes = [500]


if DATASET == 'fashion_mnist':
    results_folder = '../results/fashion_mnist'
    model_name = 'LeNet'
    training_size = 60000
    n_runss = [1000]
    epochss = [1]
    sigmas = [1.0] # sigma=1.0, C=2.1; sigma=2.0, C=1.0; sigma=3.0, C=1.0; sigma=4.0, C=0.8; 
    sample_rates = [0.001]
    lrs = [0.1]
    clips = [1.0]
    sub_training_sizes = [500]


elif DATASET == 'cifar10':
    results_folder = '../results/cifar10'
    model_name = 'ResNet18-GN'
    training_size = 50000
    n_runss = [1000]
    epochss = [200]
    sigmas = [10.0] # sigmas = [1.0, 1.5, 2.0]
    sample_rates = [0.001] # sample_rates = [512/10000, 1024/10000]
    lrs = [0.01] # lrs = [0.01, 0.05, 0.1]
    clips = [10.0] # clips = [34 for sigma=1]
    sub_training_sizes = [1000]
    

if 'train' in MODE:
    for nr, ep, sig, sr, lr, c, sts in itertools.product(n_runss, epochss, sigmas, sample_rates, lrs, clips, sub_training_sizes): 
        cmd = TRAIN_COMMAND.format(dataset=DATASET, n_runs=nr, epochs=ep, sigma=sig, sample_rate=sr, lr=lr, c=c, model_name=model_name, results_folder=results_folder, sub_training_size=sts, train_mode=TRAIN_MODE)
        print(cmd)
        subprocess.call(cmd.split())

if 'eval' in MODE:
    for nr, ep, sig, sr, lr, c, sts in itertools.product(n_runss, epochss, sigmas, sample_rates, lrs, clips, sub_training_sizes): 
        cmd = EVAL_COMMAND.format(dataset=DATASET, n_runs=nr, epochs=ep, sigma=sig, sample_rate=sr, lr=lr, c=c, model_name=model_name, results_folder=results_folder, sub_training_size=sts, train_mode=TRAIN_MODE)
        print(cmd)
        subprocess.call(cmd.split())

if 'certify' in MODE:
    for nr, ep, sig, sr, lr, c, sts in itertools.product(n_runss, epochss, sigmas, sample_rates, lrs, clips, sub_training_sizes): 
        cmd = CERTIFY_COMMAND.format(n_runs=nr, epochs=ep, sigma=sig, sample_rate=sr, lr=lr, c=c, model_name=model_name, results_folder=results_folder, training_size=training_size, sub_training_size=sts, train_mode=TRAIN_MODE, mode='certify')
        print(cmd)
        subprocess.call(cmd.split())
        
if 'plot' in MODE:
    for nr, ep, sig, sr, lr, c, sts in itertools.product(n_runss, epochss, sigmas, sample_rates, lrs, clips, sub_training_sizes): 
        cmd = CERTIFY_COMMAND.format(n_runs=nr, epochs=ep, sigma=sig, sample_rate=sr, lr=lr, c=c, model_name=model_name, results_folder=results_folder, training_size=training_size, sub_training_size=sts, train_mode=TRAIN_MODE, mode='plot')
        print(cmd)
        subprocess.call(cmd.split())

if 'ablation' in MODE:
    for nr, ep, sig, sr, lr, c, sts in itertools.product(n_runss, epochss, sigmas, sample_rates, lrs, clips, sub_training_sizes): 
        cmd = CERTIFY_COMMAND.format(n_runs=nr, epochs=ep, sigma=sig, sample_rate=sr, lr=lr, c=c, model_name=model_name, results_folder=results_folder, training_size=training_size, sub_training_size=sts, train_mode=TRAIN_MODE, mode='ablation')
        print(cmd)
        subprocess.call(cmd.split())

if 'sub-acc-test' in MODE:
    for nr, ep, sig, sr, lr, c, sts in itertools.product(n_runss, epochss, sigmas, sample_rates, lrs, clips, sub_training_sizes): 
        cmd = TRAIN_SUBSET_ACC.format(dataset=DATASET, n_runs=nr, epochs=ep, sigma=sig, sample_rate=sr, lr=lr, c=c, model_name=model_name, results_folder=results_folder, sub_training_size=sts, train_mode=TRAIN_MODE)
        print(cmd)
        subprocess.call(cmd.split())

if 'summary' in MODE:
    summarys = []
    for nr, ep, sig, sr, lr, c, sts in itertools.product(n_runss, epochss, sigmas, sample_rates, lrs, clips, sub_training_sizes):
        dir_path = get_dir(TRAIN_MODE, results_folder, model_name, lr, sig, c, sr, ep, nr, sts)
        acc, eps = extract_summary(dir_path)
        summarys.append((dir_path, acc, eps))
    summarys.sort(key=lambda x:x[1])
    
    # make folder 
    summary_folder = f'{results_folder}/summary_{datetime.now()}'
    Path(summary_folder).mkdir(parents=True, exist_ok=True)
    with open(f'{summary_folder}/summary_{datetime.now()}.txt', 'w') as f:
        for line in summarys:
            f.write(str(line) + '\n')
    # move dirs
    import shutil
    for nr, ep, sig, sr, lr, c, sts in itertools.product(n_runss, epochss, sigmas, sample_rates, lrs, clips, sub_training_sizes):
        ori_dir_path = get_dir(TRAIN_MODE, results_folder, model_name, lr, sig, c, sr, ep, nr, sts)
        tar_dir_path = ori_dir_path.replace(results_folder, summary_folder)
        shutil.move(ori_dir_path, tar_dir_path)


if 'merge' in MODE:
    s_results_folders = ['../results_2/cifar10', '../results_3/cifar10']
    t_results_folder = '../results_merge/cifar10'
    acc_threshold = 0.5

    for nr, ep, sig, sr, lr, c, sts in itertools.product(n_runss, epochss, sigmas, sample_rates, lrs, clips, sub_training_sizes):
        source_folder_list = [get_dir(TRAIN_MODE, rf, model_name, lr, sig, c, sr, ep, nr, sts) for rf in s_results_folders]
        target_folder = get_dir(TRAIN_MODE, t_results_folder, model_name, lr, sig, c, sr, ep, nr, sts)
        Path(f'{target_folder}/models').mkdir(parents=True, exist_ok=True)
        merge_models(source_folder_list, target_folder, acc_threshold, nr)
        
        
    


# NOTIFIER.notify(socket.gethostname(), 'Job Done.')