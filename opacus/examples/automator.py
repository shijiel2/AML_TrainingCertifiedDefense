import sys
sys.path.append("../..")

import itertools
import socket
from subprocess import Popen
from notification import NOTIFIER

MNIST_TRAIN_COMMAND = 'python mnist.py --n-runs {n_runs} --epochs {epochs} --sigma {sigma} --sample-rate {sample_rate} --lr {lr} --save-model'
MNIST_CERTIFY_COMMAND = 'python certify.py --n-runs {n_runs} --epochs {epochs} --sigma {sigma} --sample-rate {sample_rate} --lr {lr}'

MODE = 'certify'

n_runss = [1000]
epochss = [1] 
sigmas = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5 ,5]
sample_rates = [0.001, 0.0001, 0.01]
lrs = [0.1, 0.01]

for nr, ep, sig, sr, lr in itertools.product(n_runss, epochss, sigmas, sample_rates, lrs):
    if MODE == 'train':
        proc = Popen(MNIST_TRAIN_COMMAND.format(n_runs=nr, epochs=ep, sigma=sig, sample_rate=sr, lr=lr),
                        shell=True,
                        cwd='./')
        proc.wait()
    elif MODE == 'certify':
        print(MNIST_CERTIFY_COMMAND.format(n_runs=nr, epochs=ep, sigma=sig, sample_rate=sr, lr=lr))
        proc = Popen(MNIST_CERTIFY_COMMAND.format(n_runs=nr, epochs=ep, sigma=sig, sample_rate=sr, lr=lr),
                        shell=True,
                        cwd='./')
        proc.wait()
        break

NOTIFIER.notify(socket.gethostname(), 'Job Done.')