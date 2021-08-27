import itertools
import socket
import sys
from pandas.core.base import DataError

from torch.utils.data import dataset
sys.path.append("../..")
from subprocess import Popen

from notification import NOTIFIER


MODE = ['train', 'certify', 'plot', 'neval', 'nsub-acc-test']
DATASET = 'mnist'
TRAIN_MODE = 'Sub-DP-no-RDP-amp' # DP, Sub-DP, Bagging, Sub-DP-no-RDP-amp


TRAIN_COMMAND = 'python {dataset}.py --n-runs {n_runs} --epochs {epochs} --sigma {sigma} --sample-rate {sample_rate} --lr {lr} -c {c} --sub-training-size {sub_training_size} --save-model --train-mode {train_mode}'

EVAL_COMMAND = 'python {dataset}.py --n-runs {n_runs} --epochs {epochs} --sigma {sigma} --sample-rate {sample_rate} --lr {lr} -c {c} --sub-training-size {sub_training_size} --load-model --train-mode {train_mode}'

CERTIFY_COMMAND = 'python certify.py --n-runs {n_runs} --epochs {epochs} --sigma {sigma} --sample-rate {sample_rate} --lr {lr} -c {c} --model-name {model_name} --results-folder {results_folder} --training-size {training_size} --sub-training-size {sub_training_size} --train-mode {train_mode}'

PLOT_COMMAND = 'python certify.py --n-runs {n_runs} --epochs {epochs} --sigma {sigma} --sample-rate {sample_rate} --lr {lr} -c {c} --model-name {model_name} --results-folder {results_folder} --training-size {training_size} --sub-training-size {sub_training_size} --train-mode {train_mode} --plot'

TRAIN_SUBSET_ACC = 'python {dataset}.py --n-runs {n_runs} --epochs {epochs} --sigma {sigma} --sample-rate {sample_rate} --lr {lr} -c {c} --sub-training-size {sub_training_size} --save-model --train-mode {train_mode} --sub-acc-test'

if DATASET == 'mnist':
    results_folder = '../results/mnist'
    model_name = 'SampleConvNet'
    training_size = 60000
    n_runss = [1000]
    epochss = [1]
    sigmas = [2]
    sample_rates = [0.001]
    lrs = [0.1]
    clips = [1]
    sub_training_sizes = [30000]

elif DATASET == 'cifar10':
    results_folder = '../results/cifar10'
    model_name = 'ConvNet'
    training_size = 50000
    n_runss = [1]
    epochss = [100]
    sigmas = [1.0, 2.0, 3.0, 4.0]
    sample_rates = [0.01/2] # 0.01 
    lrs = [0.01]
    clips = [25]
    sub_training_sizes = [25000]
    

if 'train' in MODE:
    for nr, ep, sig, sr, lr, c, sts in itertools.product(n_runss, epochss, sigmas, sample_rates, lrs, clips, sub_training_sizes): 
        print(TRAIN_COMMAND.format(dataset=DATASET, n_runs=nr, epochs=ep, sigma=sig, sample_rate=sr, lr=lr, c=c, sub_training_size=sts, train_mode=TRAIN_MODE))
        proc = Popen(TRAIN_COMMAND.format(dataset=DATASET, n_runs=nr, epochs=ep, sigma=sig, sample_rate=sr, lr=lr, c=c, sub_training_size=sts, train_mode=TRAIN_MODE),
                        shell=True,
                        cwd='./')
        proc.wait()

if 'certify' in MODE:
    for nr, ep, sig, sr, lr, c, sts in itertools.product(n_runss, epochss, sigmas, sample_rates, lrs, clips, sub_training_sizes): 
        print(CERTIFY_COMMAND.format(n_runs=nr, epochs=ep, sigma=sig, sample_rate=sr, lr=lr, c=c, model_name=model_name, results_folder=results_folder, training_size=training_size, sub_training_size=sts, train_mode=TRAIN_MODE))
        proc = Popen(CERTIFY_COMMAND.format(n_runs=nr, epochs=ep, sigma=sig, sample_rate=sr, lr=lr, c=c, model_name=model_name, results_folder=results_folder, training_size=training_size, sub_training_size=sts, train_mode=TRAIN_MODE),
                        shell=True,
                        cwd='./')
        proc.wait()

if 'plot' in MODE:
    for nr, ep, sig, sr, lr, c, sts in itertools.product(n_runss, epochss, sigmas, sample_rates, lrs, clips, sub_training_sizes): 
        print(PLOT_COMMAND.format(n_runs=nr, epochs=ep, sigma=sig, sample_rate=sr, lr=lr, c=c, model_name=model_name, results_folder=results_folder, training_size=training_size, sub_training_size=sts, train_mode=TRAIN_MODE))
        proc = Popen(PLOT_COMMAND.format(n_runs=nr, epochs=ep, sigma=sig, sample_rate=sr, lr=lr, c=c, model_name=model_name, results_folder=results_folder, training_size=training_size, sub_training_size=sts, train_mode=TRAIN_MODE),
                        shell=True,
                        cwd='./')
        proc.wait()

if 'eval' in MODE:
    for nr, ep, sig, sr, lr, c, sts in itertools.product(n_runss, epochss, sigmas, sample_rates, lrs, clips, sub_training_sizes): 
        print(EVAL_COMMAND.format(dataset=DATASET, n_runs=nr, epochs=ep, sigma=sig, sample_rate=sr, lr=lr, c=c, sub_training_size=sts, train_mode=TRAIN_MODE))
        proc = Popen(EVAL_COMMAND.format(dataset=DATASET, n_runs=nr, epochs=ep, sigma=sig, sample_rate=sr, lr=lr, c=c, sub_training_size=sts, train_mode=TRAIN_MODE),
                        shell=True,
                        cwd='./')
        proc.wait()

if 'sub-acc-test' in MODE:
    for nr, ep, sig, sr, lr, c, sts in itertools.product(n_runss, epochss, sigmas, sample_rates, lrs, clips, sub_training_sizes): 
        print(TRAIN_SUBSET_ACC.format(dataset=DATASET, n_runs=nr, epochs=ep, sigma=sig, sample_rate=sr, lr=lr, c=c, sub_training_size=sts, train_mode=TRAIN_MODE))
        proc = Popen(TRAIN_SUBSET_ACC.format(dataset=DATASET, n_runs=nr, epochs=ep, sigma=sig, sample_rate=sr, lr=lr, c=c, sub_training_size=sts, train_mode=TRAIN_MODE),
                        shell=True,
                        cwd='./')
        proc.wait()

# proc = Popen(MNIST_CERTIFY_COMMAND.format(n_runs=1000, epochs=1, sigma=3.5, sample_rate=0.001, lr=0.1),
#              shell=True,
#              cwd='./')
# proc.wait()

NOTIFIER.notify(socket.gethostname(), 'Job Done.')