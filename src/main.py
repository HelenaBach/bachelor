import os
import sys
import cv2
import segmentation
import matplotlib.pyplot as plt
import parser
from table import image_table
import aligner
import pca
import asm
import knn
import math
import pickle 
import random # test knn 
import numpy as np

try:
        # get the name of the csv file of the reviews
        path = sys.argv[1] 
except:
        print('The path of the training image directory should be passed as argument to this script')
        sys.exit(2)

try:
        # get the name of the csv file of the reviews
        path_test = sys.argv[2] 
except:
        print('The path of the test image directory should be passed as argument to this script')
        sys.exit(2)

# get all images
images = os.listdir(path)[:1200]

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

# update feature vectors in image table
# returns the mean shape, the principal axis and
# a tuple of (variance, percentage of variance)
mean, var_matrix, principal_axis, components = asm.construct()

asm_model = mean, var_matrix, principal_axis, components

# return list of feature vectores from image table
training_data = knn.construct()


# get all images
test_list = os.listdir(path_test)
random.shuffle(test_list)

test_images = test_list[:20]

k = 5
# initialize the image table 
for test_image in test_images:
    # make sure we only test each image one time
    if test_image.endswith('.jpg'):
        # remove the ending of the image
        test_image = test_image[:-4]
        
        gray_image = parser.get_grayscale(path_test, test_image)
        # get image features
        
        # return a feature vector
        # TESTER DET ER DUMT
        # gør Otsu agtige ting
        binary = segmentation.otsu(gray_image)
        # set landmarks
        landmarks_temp = segmentation.landmark_setter(binary, gray_image)
        x = aligner.solve_x(mean, landmarks_temp, var_matrix)
        aligned_landmarks = aligner.align_pair(landmarks_temp, x)
        image_features = np.dot(principal_axis, aligned_landmarks-mean)
        # NU TESTER VI IKKE MERE
        #image_features = asm.image_search(asm_model, gray_image)

        # classify new image from training data 
        # get a sorted list of the class id and the number of votes
        label_candidates = knn.classify(training_data, image_features, k)

        # label_candidates[0][0] should give the classification 
        label = label_candidates[0][0]

        # check if the classification was right and
        # stats.do_shit(test_image, label)
        print('test image:')
        print(test_image)
        print('label:')
        print(label)


