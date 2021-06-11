import sys
sys.path.append("../..")

import socket
from subprocess import Popen
from notification import NOTIFIER

MNIST_COMMAND = 'python mnist.py --n-runs {n_runs} --epochs {epochs} --sigma {sigma} --sample-rate {sample_rate} --lr {lr} --save-model'

n_runss = [1000]
epochss = [1] 
sigmas = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5 ,5]
sample_rates = [0.001, 0.0001, 0.01]
lrs = [0.1, 0.01]

for nr, ep, sig, sr, lr in zip(n_runss, epochss, sigmas, sample_rates, lrs):
    proc = Popen(MNIST_COMMAND.format(n_runs=nr, epochs=ep, sigma=sig, sample_rate=sr, lr=lr),
                    shell=True,
                    cwd='./')
    proc.wait()

NOTIFIER.notify(socket.gethostname(), 'Job Done.')