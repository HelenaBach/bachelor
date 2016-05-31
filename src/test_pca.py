import cv2
import numpy as np
import math
from table import image_table
import sys # debugging
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

import aligner
import pca

#with open(im_table, 'rb') as f:
#    img_table = pickle.load(f)

with open('mean_more.p', 'rb') as f:
    mean = pickle.load(f)

with open('image_table_more.p', 'rb') as f:
    image_table = pickle.load(f)

with open('image_table_unaligned.p', 'rb') as f:
    image_table_unaligned = pickle.load(f)

# plot the mean shape
xes = mean[::2]
yes = mean[1::2]
plt.plot(xes,yes)
plt.show()

# plot 5 unaligned shapes
for img in image_table_unaligned[:10]:
    shape = img['landmarks']
    un_xes = shape[::2]
    un_yes = shape[1::2]
    plt.plot(un_xes, un_yes)

plt.show()

#plot 5 aligned shapes
for img in image_table[:10]:
    shape = img['landmarks']
    notun_xes = shape[::2]
    notun_yes = shape[1::2]
    plt.plot(notun_xes, notun_yes)

plt.show()
#sys.exit(2)
# plot PCA variance and stuff
data = []
for im_struct in image_table:
    landmarks = im_struct['landmarks']
    data.append(landmarks)

principal_axis, comp_variance = pca.fit(data, mean, 200)

accum_var = 0
accum_xes = []
accum_yes = []
for i in range(200):
	accum_xes.append(i)
	accum_yes.append(accum_var)
	accum_var += comp_variance[i][1]
accum_xes.append(200)
accum_yes.append(accum_var)
plt.plot(accum_xes,accum_yes)
plt.show()
sys.exit(2)

x = []
y = []
z = []
for im_struct in image_table:
    # mean centred shape
    shape = im_struct['landmarks'] - mean

    # translate data into PCA space
    im_struct['feature_vector'] = np.dot(principal_axis, shape)
    x.append(im_struct['feature_vector'][0])
    y.append(im_struct['feature_vector'][1])
    z.append(im_struct['feature_vector'][2])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, marker='.')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
sys.exit(2)
for j in range(len(principal_axis)):
    plt.plot(xes, yes)
    print(math.sqrt(comp_variance[j][0]))
    for i in range(3):
        feat_vector = principal_axis[j]*(math.sqrt(comp_variance[j][0])*(i+1))
        feat_xes = feat_vector[::2]
        feat_yes = feat_vector[1::2]
        new_xes_p = xes + feat_xes
        new_yes_p = yes + feat_yes
        new_xes_m = xes - feat_xes
        new_yes_m = yes - feat_yes

        plt.plot(new_xes_p, new_yes_p)
        plt.plot(new_xes_m, new_yes_m)

    plt.show()
