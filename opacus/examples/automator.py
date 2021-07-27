import itertools
import socket
import sys
from pandas.core.base import DataError

from torch.utils.data import dataset
sys.path.append("../..")
from subprocess import Popen

from notification import NOTIFIER


MODE = ['train', 'certify', 'nplot', 'neval']
DATASET = 'cifar10'


TRAIN_COMMAND = 'python {dataset}.py --n-runs {n_runs} --epochs {epochs} --sigma {sigma} --sample-rate {sample_rate} --lr {lr} -c {c} --save-model'
TRAIN_BAGGING_COMMAND = 'python {dataset}.py --n-runs {n_runs} --epochs {epochs} --lr {lr} -c {c} --bagging-size {bagging_size} --disable-dp --save-model'

EVAL_COMMAND = 'python {dataset}.py --n-runs {n_runs} --epochs {epochs} --sigma {sigma} --sample-rate {sample_rate} --lr {lr} -c {c} --load-model'

CERTIFY_COMMAND = 'python certify.py --n-runs {n_runs} --epochs {epochs} --sigma {sigma} --sample-rate {sample_rate} --lr {lr} -c {c} --model-name {model_name} --results-folder {results_folder} --method-name DP'
CERTIFY_BAGGING_COMMAND = 'python certify.py --n-runs {n_runs} --epochs {epochs} --lr {lr} -c {c} --training-size {training_size} --bagging-size {bagging_size} --model-name {model_name} --results-folder {results_folder} --disable-dp --method-name Bagging'

PLOT_COMMAND = 'python certify.py --n-runs {n_runs} --epochs {epochs} --sigma {sigma} --sample-rate {sample_rate} --lr {lr} -c {c} --model-name {model_name} --results-folder {results_folder} --plot --method-name DP'
PLOT_BAGGING_COMMAND = 'python certify.py --n-runs {n_runs} --epochs {epochs} --lr {lr} -c {c} --training-size {training_size} --bagging-size {bagging_size} --model-name {model_name} --results-folder {results_folder} --disable-dp --plot --method-name Bagging'
PLOT_DPBASELINE_COMMAND = 'python certify.py --n-runs {n_runs} --epochs {epochs} --sigma {sigma} --sample-rate {sample_rate} --lr {lr} -c {c} --model-name {model_name} --results-folder {results_folder} --plot --method-name DP-Baseline'
PLOT_EPOCH_ACC_EPS_COMMAND = 'python certify.py --n-runs {n_runs} --epochs {epochs} --sigma {sigma} --sample-rate {sample_rate} --lr {lr} -c {c} --model-name {model_name} --results-folder {results_folder} --plot --method-name Epoch_acc_eps'
PLOT_SUBSET_ACC_COMMAND = 'python certify.py --n-runs {n_runs} --epochs {epochs} --sigma {sigma} --sample-rate {sample_rate} --lr {lr} -c {c} --model-name {model_name} --results-folder {results_folder} --plot --method-name Subset_acc'

if DATASET == 'mnist':
    results_folder = '../results/mnist'
    model_name = 'SampleConvNet'
    training_size = 60000
    n_runss = [1000]
    epochss = [1]
    sigmas = [0.5 * x for x in range(2, 11)]
    sample_rates = [0.001]
    lrs = [0.1]
    clips = [1]
    bagging_sizes = [900]
    bagging_epochss = [5]

elif DATASET == 'cifar10':
    results_folder = '../results/cifar10'
    model_name = 'ConvNet'
    training_size = 50000
    n_runss = [1000]
    epochss = [20]
    sigmas = [1.0]
    sample_rates = [0.01]
    lrs = [0.01]
    clips = [25]
    bagging_sizes = [5000]
    bagging_epochss = [50]
    
    


if 'train' in MODE:
    # DP models
    for nr, ep, sig, sr, lr, c in itertools.product(n_runss, epochss, sigmas, sample_rates, lrs, clips): 
        print(TRAIN_COMMAND.format(dataset=DATASET, n_runs=nr, epochs=ep, sigma=sig, sample_rate=sr, lr=lr, c=c))
        proc = Popen(TRAIN_COMMAND.format(dataset=DATASET, n_runs=nr, epochs=ep, sigma=sig, sample_rate=sr, lr=lr, c=c),
                        shell=True,
                        cwd='./')
        proc.wait()

    # # Bagging models
    # for nr, ep, lr, bs, c in itertools.product(n_runss, bagging_epochss, lrs, bagging_sizes, clips):
    #     print(TRAIN_BAGGING_COMMAND.format(dataset=DATASET, n_runs=nr, epochs=ep, lr=lr, c=c, bagging_size=bs))
    #     proc = Popen(TRAIN_BAGGING_COMMAND.format(dataset=DATASET, n_runs=nr, epochs=ep, lr=lr, c=c, bagging_size=bs),
    #                     shell=True,
    #                     cwd='./')
    #     proc.wait()

if 'certify' in MODE:
    # DP models
    for nr, ep, sig, sr, lr, c in itertools.product(n_runss, epochss, sigmas, sample_rates, lrs, clips): 
        print(CERTIFY_COMMAND.format(
            n_runs=nr, epochs=ep, sigma=sig, sample_rate=sr, lr=lr, c=c, model_name=model_name, results_folder=results_folder))
        proc = Popen(CERTIFY_COMMAND.format(n_runs=nr, epochs=ep, sigma=sig, sample_rate=sr, lr=lr, c=c, model_name=model_name, results_folder=results_folder),
                        shell=True,
                        cwd='./')
        proc.wait()

    # # Bagging models
    # for nr, ep, lr, bs, c in itertools.product(n_runss, bagging_epochss, lrs, bagging_sizes, clips):
    #     print(CERTIFY_BAGGING_COMMAND.format(n_runs=nr, epochs=ep, lr=lr, c=c, bagging_size=bs, model_name=model_name, results_folder=results_folder, training_size=training_size))
    #     proc = Popen(CERTIFY_BAGGING_COMMAND.format(n_runs=nr, epochs=ep, lr=lr, c=c, bagging_size=bs, model_name=model_name, results_folder=results_folder, training_size=training_size),
    #                     shell=True,
    #                     cwd='./')
    #     proc.wait()

if 'plot' in MODE:
    # DP models
    for nr, ep, sig, sr, lr, c in itertools.product(n_runss, epochss, sigmas, sample_rates, lrs, clips): 
        print(PLOT_COMMAND.format(
            n_runs=nr, epochs=ep, sigma=sig, sample_rate=sr, lr=lr, c=c, model_name=model_name, results_folder=results_folder))
        proc = Popen(PLOT_COMMAND.format(n_runs=nr, epochs=ep, sigma=sig, sample_rate=sr, lr=lr, c=c, model_name=model_name, results_folder=results_folder),
                        shell=True,
                        cwd='./')
        proc.wait()

    # # Bagging models
    # for nr, ep, lr, bs, c in itertools.product(n_runss, bagging_epochss, lrs, bagging_sizes, clips):
    #     print(PLOT_BAGGING_COMMAND.format(n_runs=nr, epochs=ep, lr=lr, c=c, bagging_size=bs, model_name=model_name, results_folder=results_folder, training_size=training_size))
    #     proc = Popen(PLOT_BAGGING_COMMAND.format(n_runs=nr, epochs=ep, lr=lr, c=c, bagging_size=bs, model_name=model_name, results_folder=results_folder, training_size=training_size),
    #                     shell=True,
    #                     cwd='./')
    #     proc.wait()

    # # DP-Baseline models
    # for nr, ep, sig, sr, lr, c in itertools.product(n_runss, epochss, sigmas, sample_rates, lrs, clips): 
    #     print(PLOT_DPBASELINE_COMMAND.format(
    #         n_runs=nr, epochs=ep, sigma=sig, sample_rate=sr, lr=lr, c=c, model_name=model_name, results_folder=results_folder))
    #     proc = Popen(PLOT_DPBASELINE_COMMAND.format(n_runs=nr, epochs=ep, sigma=sig, sample_rate=sr, lr=lr, c=c, model_name=model_name, results_folder=results_folder),
    #                     shell=True,
    #                     cwd='./')
    #     proc.wait()

    # # Epoch_acc_eps
    # for nr, ep, sig, sr, lr, c in itertools.product(n_runss, epochss, sigmas, sample_rates, lrs, clips): 
    #     print(PLOT_EPOCH_ACC_EPS_COMMAND.format(
    #         n_runs=nr, epochs=ep, sigma=sig, sample_rate=sr, lr=lr, c=c, model_name=model_name, results_folder=results_folder))
    #     proc = Popen(PLOT_EPOCH_ACC_EPS_COMMAND.format(n_runs=nr, epochs=ep, sigma=sig, sample_rate=sr, lr=lr, c=c, model_name=model_name, results_folder=results_folder),
    #                     shell=True,
    #                     cwd='./')
    #     proc.wait()

    # Subset_acc
    for nr, ep, sig, sr, lr, c in itertools.product(n_runss, epochss, sigmas, sample_rates, lrs, clips): 
        print(PLOT_SUBSET_ACC_COMMAND.format(
            n_runs=nr, epochs=ep, sigma=sig, sample_rate=sr, lr=lr, c=c, model_name=model_name, results_folder=results_folder))
        proc = Popen(PLOT_SUBSET_ACC_COMMAND.format(n_runs=nr, epochs=ep, sigma=sig, sample_rate=sr, lr=lr, c=c, model_name=model_name, results_folder=results_folder),
                        shell=True,
                        cwd='./')
        proc.wait()


if 'eval' in MODE:
    # DP models
    for nr, ep, sig, sr, lr, c in itertools.product(n_runss, epochss, sigmas, sample_rates, lrs, clips): 
        print(EVAL_COMMAND.format(dataset=DATASET, n_runs=nr, epochs=ep, sigma=sig, sample_rate=sr, lr=lr, c=c))
        proc = Popen(EVAL_COMMAND.format(dataset=DATASET, n_runs=nr, epochs=ep, sigma=sig, sample_rate=sr, lr=lr, c=c),
                        shell=True,
                        cwd='./')
        proc.wait()

# proc = Popen(MNIST_CERTIFY_COMMAND.format(n_runs=1000, epochs=1, sigma=3.5, sample_rate=0.001, lr=0.1),
#              shell=True,
#              cwd='./')
# proc.wait()

NOTIFIER.notify(socket.gethostname(), 'Job Done.')