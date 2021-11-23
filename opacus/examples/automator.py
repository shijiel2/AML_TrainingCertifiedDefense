import itertools
import socket
import sys

sys.path.append("../..")
import subprocess
from certify_utilis import get_dir, extract_summary
from notification import NOTIFIER
from datetime import datetime


MODE = ['ntrain', 'ncertify', 'nplot', 'ablation', 'neval', 'nsub-acc-test', 'nsummary']
DATASET = 'cifar10'
TRAIN_MODE = 'Sub-DP-no-amp' # DP, Sub-DP, Bagging, Sub-DP-no-amp


TRAIN_COMMAND = 'python {dataset}.py --n-runs {n_runs} --epochs {epochs} --sigma {sigma} --sample-rate {sample_rate} --lr {lr} -c {c} --model-name {model_name} --sub-training-size {sub_training_size} --train-mode {train_mode} --save-model' # --save-model

EVAL_COMMAND = 'python {dataset}.py --n-runs {n_runs} --epochs {epochs} --sigma {sigma} --sample-rate {sample_rate} --lr {lr} -c {c} --model-name {model_name} --sub-training-size {sub_training_size} --train-mode {train_mode} --load-model'

CERTIFY_COMMAND = 'python certify.py --n-runs {n_runs} --epochs {epochs} --sigma {sigma} --sample-rate {sample_rate} --lr {lr} -c {c} --model-name {model_name} --results-folder {results_folder} --training-size {training_size} --sub-training-size {sub_training_size} --train-mode {train_mode} --mode {mode}'

TRAIN_SUBSET_ACC = 'python {dataset}.py --n-runs {n_runs} --epochs {epochs} --sigma {sigma} --sample-rate {sample_rate} --lr {lr} -c {c} --model-name {model_name} --sub-training-size {sub_training_size} --save-model --train-mode {train_mode} --sub-acc-test'

if DATASET == 'mnist':
    results_folder = '../results/mnist'
    model_name = 'SampleConvNet'
    training_size = 60000
    n_runss = [1000]
    epochss = [15]
    sigmas = [0.7]
    sample_rates = [32/500]
    lrs = [0.25]
    clips = [1.5]
    sub_training_sizes = [500]

elif DATASET == 'cifar10':
    results_folder = '../results/cifar10'
    model_name = 'ConvNet'
    training_size = 50000
    n_runss = [1000]
    epochss = [90]
    sigmas = [2.0] # sigmas = [1.0, 1.5, 2.0]
    sample_rates = [1024/100000] # sample_rates = [512/10000, 1024/10000]
    lrs = [0.01] # lrs = [0.01, 0.05, 0.1]
    clips = [20.0] # clips = [20.0, 25.0, 30.0]
    sub_training_sizes = [10000]
    

if 'train' in MODE:
    for nr, ep, sig, sr, lr, c, sts in itertools.product(n_runss, epochss, sigmas, sample_rates, lrs, clips, sub_training_sizes): 
        cmd = TRAIN_COMMAND.format(dataset=DATASET, n_runs=nr, epochs=ep, sigma=sig, sample_rate=sr, lr=lr, c=c, model_name=model_name, sub_training_size=sts, train_mode=TRAIN_MODE)
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

if 'eval' in MODE:
    for nr, ep, sig, sr, lr, c, sts in itertools.product(n_runss, epochss, sigmas, sample_rates, lrs, clips, sub_training_sizes): 
        cmd = EVAL_COMMAND.format(dataset=DATASET, n_runs=nr, epochs=ep, sigma=sig, sample_rate=sr, lr=lr, c=c, model_name=model_name, sub_training_size=sts, train_mode=TRAIN_MODE)
        print(cmd)
        subprocess.call(cmd.split())

if 'sub-acc-test' in MODE:
    for nr, ep, sig, sr, lr, c, sts in itertools.product(n_runss, epochss, sigmas, sample_rates, lrs, clips, sub_training_sizes): 
        cmd = TRAIN_SUBSET_ACC.format(dataset=DATASET, n_runs=nr, epochs=ep, sigma=sig, sample_rate=sr, lr=lr, c=c, model_name=model_name, sub_training_size=sts, train_mode=TRAIN_MODE)
        print(cmd)
        subprocess.call(cmd.split())

if 'summary' in MODE:
    summarys = []
    for nr, ep, sig, sr, lr, c, sts in itertools.product(n_runss, epochss, sigmas, sample_rates, lrs, clips, sub_training_sizes):
        dir_path = get_dir(TRAIN_MODE, results_folder, model_name, lr, sig, c, sr, ep, nr, sts)
        trainlog_path = f'{dir_path}/train.log'
        with open(trainlog_path, 'r') as f:
            lines = f.readlines()
            acc, eps = extract_summary(lines)
            summarys.append((dir_path, acc, eps))
    summarys.sort(key=lambda x:x[1])
    with open(f'{results_folder}/summary_{datetime.now()}.txt', 'w') as f:
        for line in summarys:
            f.write(str(line) + '\n')


# NOTIFIER.notify(socket.gethostname(), 'Job Done.')