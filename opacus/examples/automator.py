import itertools
import socket
import sys
sys.path.append("../..")
from subprocess import Popen

from notification import NOTIFIER

MNIST_TRAIN_COMMAND = 'python mnist.py --n-runs {n_runs} --epochs {epochs} --sigma {sigma} --sample-rate {sample_rate} --lr {lr} --save-model'
MNIST_CERTIFY_COMMAND = 'python certify.py --n-runs {n_runs} --epochs {epochs} --sigma {sigma} --sample-rate {sample_rate} --lr {lr}'
MNIST_PLOT_COMMAND = 'python certify.py --n-runs {n_runs} --epochs {epochs} --sigma {sigma} --sample-rate {sample_rate} --lr {lr} --plot'

MODE = 'certify'

n_runss = [1000]
epochss = [1]
# sigmas = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
sigmas = [5]
sample_rates = [0.001]
lrs = [0.1]

for nr, ep, sig, sr, lr in itertools.product(n_runss, epochss, sigmas, sample_rates, lrs):
    if MODE == 'train':
        proc = Popen(MNIST_TRAIN_COMMAND.format(n_runs=nr, epochs=ep, sigma=sig, sample_rate=sr, lr=lr),
                     shell=True,
                     cwd='./')
        proc.wait()
    elif MODE == 'certify':
        print(MNIST_CERTIFY_COMMAND.format(
            n_runs=nr, epochs=ep, sigma=sig, sample_rate=sr, lr=lr))
        proc = Popen(MNIST_CERTIFY_COMMAND.format(n_runs=nr, epochs=ep, sigma=sig, sample_rate=sr, lr=lr),
                     shell=True,
                     cwd='./')
        proc.wait()
    elif MODE == 'plot':
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