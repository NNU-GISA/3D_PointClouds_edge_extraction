n1=6

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import h5py

BASE_DIR = os.path.dirname(os.path.abspath('/home/rahulchakwate/My_tensorflow/3D_Object_Segmentation/PointNet_Implementation/data'))
sys.path.append(BASE_DIR)


def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

TRAIN_FILES = getDataFiles( \
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))

def loadDataFile(filename):
    return load_h5(filename)

LOG_FOUT = open(os.path.join(BASE_DIR, 'log_train.txt'), 'w')

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

train_file_idxs = np.arange(0, len(TRAIN_FILES))

NUM_POINT=1024
log_string('----' + str(0) + '-----')
current_data, current_label = loadDataFile(TRAIN_FILES[train_file_idxs[0]])
current_data=current_data[:,0:NUM_POINT,:]
print(current_data.shape)
print(current_label.shape)


#plotting
xs=current_data[n1:n1+4,:,0]
ys=current_data[n1:n1+4,:,1]
zs=current_data[n1:n1+4,:,2]
print(xs.shape)
fig1 = plt.figure()
fig4 = plt.figure()
fig3 = plt.figure()
fig2 = plt.figure()
ax = fig1.add_subplot(111, projection='3d')
bx = fig2.add_subplot(121, projection='3d')
cx = fig3.add_subplot(211, projection='3d')
dx = fig4.add_subplot(221, projection='3d')



# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
for c, m, zlow, zhigh in [('r', '*', -50, -25), ('b', '^', -30, -5)]:
   
    ax.scatter(xs[0], ys[0], zs[0], c='b', marker=m)
    bx.scatter(xs[1], ys[1], zs[1], c='b', marker=m)
    cx.scatter(xs[2], ys[2], zs[2], c='b', marker=m)
    dx.scatter(xs[3], ys[3], zs[3], c='b', marker=m)


ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

