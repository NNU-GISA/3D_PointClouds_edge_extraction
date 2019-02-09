
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import h5py
import random
import math
import operator

BASE_DIR = os.path.dirname(os.path.abspath('/home/rahulchakwate/My_tensorflow/3D Object Segmentation/PointNet Implementation/data'))
sys.path.append(BASE_DIR)

##load function
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




##edge extraction

def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)
 
def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

def cov(x,y):
    xm=0
    ym=0
    sum=0
    for i in range(len(x)):
        xm+=x[i]
        ym+=y[i]
    xm/= (len(x))
    #print(xm)
    ym/= (len(y))
    for i in range(len(x)):
        sum+= (x[i]-xm)*(y[i]-ym)
        covxy = sum/(len(x)-1)
    return covxy


def compute_surface_var(all_pts):
    app_pts=np.empty([len(all_pts),4])
    for i in range(len(all_pts)):
        mean_pt=all_pts[i]
        #print(all_pts.shape)
        #print(mean_pt.shape)
        qrs = getNeighbors(all_pts, mean_pt, k)
        neighbors=np.array(qrs)
        #print(neighbors.shape)
        xn=neighbors[:,0]
        yn=neighbors[:,1]
        zn=neighbors[:,2]
        
        cov_list=[[cov(xn,xn), cov(xn,yn),cov(xn,zn)],[cov(yn,xn), cov(yn,yn),cov(yn,zn)],[cov(zn,xn), cov(zn,yn), cov(zn,zn)]]
        cov_matrix=np.array(cov_list)
        #print(cov_matrix)

        w,v = np.linalg.eig(cov_matrix)
        #print(w)
        eig_val = np.sort(w)
        #print(eig_val)

        surface_var = eig_val[0] / (eig_val[0]+eig_val[1]+eig_val[2])
        app_pts[i]=np.append(all_pts[i],surface_var)
        #print(surface_var)
        #print(mean_pt)
    return app_pts


#variables
NUM_POINTS= 2048
k=20


log_string('----' + str(0) + '-----')
current_data, current_label = loadDataFile(TRAIN_FILES[train_file_idxs[0]])
current_data=current_data[:,0:NUM_POINTS,:]
#print(current_data.shape)
#print(current_label.shape)
all_pts=current_data[7]
print(all_pts.shape)
all_pts=compute_surface_var(all_pts)
print(all_pts.shape)
max_sur_var=np.amax(all_pts[:,3])
min_sur_var=np.amin(all_pts[:,3])
th1=(0.3*max_sur_var + 0.7*min_sur_var)
for i in range(len(all_pts)):
    if all_pts[i,3]>th1:
        all_pts[i,3]=1
    else:
        all_pts[i,3]=0

        



#plotting
x=all_pts[:,0]
y=all_pts[:,1]
z=all_pts[:,2]
c=all_pts[:,3]
fig1 = plt.figure()

ax = fig1.add_subplot(111, projection='3d')



# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
#for c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
   
my_plot = ax.scatter(x, y, z, c=c,cmap='binary', marker='o', s=30)
#ax.scatter(xn, yn, zn, c='b', marker='D')

cbar=fig1.colorbar(my_plot)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

