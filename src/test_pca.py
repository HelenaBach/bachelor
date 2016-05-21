import cv2
import numpy as np
import math
from table import image_table
import sys # debugging
import matplotlib.pyplot as plt

import pickle 

import aligner
import pca 

#with open(im_table, 'rb') as f:
#    img_table = pickle.load(f)

with open('mean.p', 'rb') as f:
    mean = pickle.load(f)

with open('data.p', 'rb') as f:
    data = pickle.load(f)

# plot the mean shape
xes = mean[::2]
yes = mean[1::2]

adjusted_data, feature_vector, comp_variance = pca.fit(data, mean, 0.90)

#for j in range(len(feature_vector)):
#	plt.plot(xes, yes)
#	for i in range(3):
#	    feat_vector = feature_vector[j]*(math.sqrt(comp_variance[j][0])*(i+1))
#	    feat_xes = feat_vector[::2]
#	    feat_yes = feat_vector[1::2]
#	    new_xes_p = xes + feat_xes
#	    new_yes_p = yes + feat_yes
#	    new_xes_m = xes - feat_xes
#	    new_yes_m = yes - feat_yes
#	
#	    plt.plot(new_xes_p, new_yes_p)
#	    plt.plot(new_xes_m, new_yes_m)
#
#	plt.show()

#from sklearn.decomposition import PCA
#
#pca = PCA(n_components=200)
#pca.fit(data)
#print(pca.explained_variance_ratio_)

