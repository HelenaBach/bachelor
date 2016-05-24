import os
import sys
import cv2
import segmentation
import matplotlib.pyplot as plt
import parser
from table import image_table
import aligner
#import pca
import math
import pickle
import time # to print time of execution

start_time = time.time()

with open('mean.p', 'rb') as f:
    mean = pickle.load(f)

with open('image_table.p', 'rb') as f:
    image_table = pickle.load(f)

with open('var_matrix.p', 'rb') as f:
    var_matrix = pickle.load(f)

shape1 = np.copy(image_table[0]['landmarks'])
# rotate, scale and translate each shape to align with the first shape
# initial previous mean_shape - dummy 1x200 vector of zeros
prev_mean = mean

for i in range(100):
    print('aligner iteration: ' + str(i))
    mean = aligner.mean_shape()
    # check if prev_mean and mean is 'equal' - does the process converge
    diff = sum(abs(prev_mean-mean))
    print('sum of diff: ' + str(diff))
    if diff < 100 or i == 99:
        print('sum of diff: ' + str(diff))
        break
    new_mean = aligner.normalize_mean(shape1, mean, var_matrix)
    aligner.align_all_shapes(new_mean)
    prev_mean = mean

with open('image_table_more.p', 'wb') as f:
        pickle.dump(image_table, f)
with open('mean_more.p', 'wb') as f:
        pickle.dump(mean, f)
with open('var_matrix_more.p', 'wb') as f:
        pickle.dump(var_matrix, f)

print("--- %s seconds ---" % (time.time() - start_time))
