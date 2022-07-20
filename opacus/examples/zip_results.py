from zipfile import ZipFile
import os

zf = ZipFile('../results.zip', 'w')

for dirname, subdirs, files in os.walk("../results"):
    if 'models' in subdirs:
        subdirs.remove('models')
    zf.write(dirname)
    for filename in files:
        zf.write(os.path.join(dirname, filename))
zf.close()