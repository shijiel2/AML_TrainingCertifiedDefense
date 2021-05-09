from subprocess import Popen
from notification import NOTIFIER
import socket

train_command = 'python train.py --k {k} --end {base_num} --poison_size {ps}'
certify_command = 'python certify.py --k {k} --ns {base_num} --poison_size {ps}'

k_list = [100, 500, 1000, 5000]
base_num = 1000
ps_list = [0]

for ps in ps_list:
    for k in k_list:
        proc = Popen(train_command.format(k=k, base_num=base_num, ps=ps),
                    shell=True,
                    cwd='./')
        proc.wait()
        proc = Popen(certify_command.format(k=k, base_num=base_num, ps=ps),
                    shell=True,
                    cwd='./')
        proc.wait()
NOTIFIER.notify(socket.gethostname(), 'Done.')