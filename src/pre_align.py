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

try:
        path = sys.argv[1]
except:
        print('The path of the image directory should be passed as argument to this script')
        sys.exit(2)

# get all images
images = os.listdir(path)

# initialize the image table
for image in images:
        if image.endswith('.xml'):
                im_struct = parser.init_image(path, image)
                image_table.append(im_struct)


for im_struct in image_table:
        # get image as grayscale
        img = parser.get_grayscale(path, im_struct['media_id'])
        # gør Otsu agtige ting
        binary = segmentation.otsu(img)
        # set landmarks
        landmarks = segmentation.landmark_setter(binary, img)
        # update the image table
        im_struct['landmarks'] = landmarks


# align the dataset
mean, var_matrix = aligner.the_real_aligner()

with open('image_table_centered_test.p', 'wb') as f:
        pickle.dump(image_table, f)
with open('mean_centered_test.p', 'wb') as f:
        pickle.dump(mean, f)
with open('var_matrix_centered_test.p', 'wb') as f:
        pickle.dump(var_matrix, f)

print("--- %s seconds ---" % (time.time() - start_time))
