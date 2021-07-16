import itertools
import socket
import sys
sys.path.append("../..")
from subprocess import Popen

from notification import NOTIFIER

MNIST_TRAIN_COMMAND = 'python mnist.py --n-runs {n_runs} --epochs {epochs} --sigma {sigma} --sample-rate {sample_rate} --lr {lr} --save-model'
MNIST_TRAIN_BAGGING_COMMAND = 'python mnist.py --n-runs {n_runs} --epochs {epochs} --lr {lr} --bagging-size {bagging_size} --disable-dp --save-model'
MNIST_CERTIFY_COMMAND = 'python certify.py --n-runs {n_runs} --epochs {epochs} --sigma {sigma} --sample-rate {sample_rate} --lr {lr}'
MNIST_PLOT_COMMAND = 'python certify.py --n-runs {n_runs} --epochs {epochs} --sigma {sigma} --sample-rate {sample_rate} --lr {lr} --plot'

MODE = 'train'

n_runss = [100]
epochss = [1]
sigmas = [1.5, 2]
sample_rates = [0.001]
lrs = [0.1]
bagging_sizes = [1500, 2000]
bagging_epochss = [6]

if MODE == 'train':
    # DP models
    for nr, ep, sig, sr, lr in itertools.product(n_runss, epochss, sigmas, sample_rates, lrs): 
        print(MNIST_TRAIN_COMMAND.format(n_runs=nr, epochs=ep, sigma=sig, sample_rate=sr, lr=lr))
        proc = Popen(MNIST_TRAIN_COMMAND.format(n_runs=nr, epochs=ep, sigma=sig, sample_rate=sr, lr=lr),
                        shell=True,
                        cwd='./')
        proc.wait()
    # Bagging models
    for nr, ep, lr, bs in itertools.product(n_runss, bagging_epochss, lrs, bagging_sizes):
        print(MNIST_TRAIN_BAGGING_COMMAND.format(n_runs=nr, epochs=5, lr=lr, bagging_size=bs))
        proc = Popen(MNIST_TRAIN_BAGGING_COMMAND.format(n_runs=nr, epochs=5, lr=lr, bagging_size=bs),
                        shell=True,
                        cwd='./')
        proc.wait()

elif MODE == 'certify':
    # DP models
    for nr, ep, sig, sr, lr in itertools.product(n_runss, epochss, sigmas, sample_rates, lrs): 
        print(MNIST_CERTIFY_COMMAND.format(
            n_runs=nr, epochs=ep, sigma=sig, sample_rate=sr, lr=lr))
        proc = Popen(MNIST_CERTIFY_COMMAND.format(n_runs=nr, epochs=ep, sigma=sig, sample_rate=sr, lr=lr),
                        shell=True,
                        cwd='./')
        proc.wait()

elif MODE == 'plot':
    # DP models
    for nr, ep, sig, sr, lr in itertools.product(n_runss, epochss, sigmas, sample_rates, lrs): 
        print(MNIST_PLOT_COMMAND.format(
            n_runs=nr, epochs=ep, sigma=sig, sample_rate=sr, lr=lr))
        proc = Popen(MNIST_PLOT_COMMAND.format(n_runs=nr, epochs=ep, sigma=sig, sample_rate=sr, lr=lr),
                        shell=True,
                        cwd='./')
        proc.wait()

# proc = Popen(MNIST_CERTIFY_COMMAND.format(n_runs=1000, epochs=1, sigma=3.5, sample_rate=0.001, lr=0.1),
#              shell=True,
#              cwd='./')
# proc.wait()

NOTIFIER.notify(socket.gethostname(), 'Job Done.')