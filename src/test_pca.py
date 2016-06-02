import cv2
import numpy as np
import math
from table import image_table
import sys # debugging
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
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

with open('pca.p', 'rb') as f:
     pca = pickle.load(f)

(principal_axis, comp_variance) = pca

# plot the mean shape
xes = mean[::2]
xes = np.append(xes, mean[0])
yes = mean[1::2]
yes = np.append(yes, mean[1])
plt.plot(xes,yes, marker='.')
plt.suptitle('Mean Shape', fontsize = 14)
axes = plt.gca()
axes.set_xlim([0,675])
axes.set_ylim([0,900])
plt.show()

# plot 5 unaligned shapes
for img in image_table_unaligned[:5]:
    shape = img['landmarks']
    un_xes = shape[::2]
    un_xes = np.append(un_xes, un_xes[0])
    un_yes = shape[1::2]
    un_yes = np.append(un_yes, un_yes[0])
    plt.plot(un_xes, un_yes)
plt.suptitle('5 Shapes', fontsize = 14)
axes = plt.gca()
axes.set_xlim([0,675])
axes.set_ylim([0,900])
plt.show()

#plot 5 aligned shapes
for img in image_table[:5]:
    shape = img['landmarks']
    notun_xes = shape[::2]
    notun_xes = np.append(notun_xes, notun_xes[0])
    notun_yes = shape[1::2]
    notun_yes = np.append(notun_yes, notun_yes[0])
    plt.plot(notun_xes, notun_yes)
axes = plt.gca()
plt.suptitle('5 Aligned Shapes', fontsize = 14)
axes.set_xlim([0,675])
axes.set_ylim([0,900])
plt.show()

sys.exit(2)
# plot PCA variance and stuff
data = []
for im_struct in image_table:
    landmarks = im_struct['landmarks']
    data.append(landmarks)

#principal_axis, comp_variance = pca.fit(data, mean, 200)
#with open('pca.p', 'wb') as f:
#    pickle.dump((principal_axis, comp_variance), f)

accum_var = 0
accum_xes = []
accum_yes = []
for i in range(200):
	accum_xes.append(i)
	accum_yes.append(accum_var)
	accum_var += comp_variance[i][1]
accum_xes.append(200)
accum_yes.append(accum_var)
fig, ax = plt.subplots(1, 1)
plt.plot(accum_xes,accum_yes)
ax.grid(zorder=0)
plt.suptitle('Accumulated Variance in Percentage', fontsize = 14)
ax.set_xlabel('Principal Components')
ax.set_ylabel('Percentage of Variance')
minor_ticks = np.arange(-5, 220, 5)                                               
#ax.set_xticks(major_ticks)                                                       
ax.set_xticks(minor_ticks, minor=True)
ax.grid(which='both')                                                            
# or if you want differnet settings for the grids:                               
ax.grid(which='minor', alpha=0.2)                                                
ax.grid(which='major', alpha=0.5)                                          
axes.set_xlim([0,220])
axes.set_ylim([0,1])
#ax.set_yticks(major_ticks)                                                       
#ax.set_yticks(minor_ticks, minor=True)   
plt.show()


x = []
y = []
z = []
'''for im_struct in image_table:
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
sys.exit(2)'''

for j in range(len(principal_axis)):
    plt.plot(xes, yes)
    print(math.sqrt(comp_variance[j][0]))
    for i in range(3):
        feat_vector = principal_axis[j]*(math.sqrt(comp_variance[j][0])*(i+1))
        feat_xes = feat_vector[::2]
        feat_xes = np.append(feat_xes, feat_xes[0])
        feat_yes = feat_vector[1::2]
        feat_yes = np.append(feat_yes, feat_yes[0])
        new_xes_p = xes + feat_xes
        new_yes_p = yes + feat_yes
        new_xes_m = xes - feat_xes
        new_yes_m = yes - feat_yes

        plt.plot(new_xes_p, new_yes_p)
        plt.plot(new_xes_m, new_yes_m)
    plt.suptitle(str(j+1) + '. Principal Component', fontsize = 14)
    axes.set_xlim([0,700])
    axes.set_ylim([0,1000])
    plt.show()
