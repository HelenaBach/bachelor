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
	path_test = sys.argv[1]
except:
	print('The path of the test image directory should be passed as argument to this script')
	sys.exit(2)

# update feature vectors in image table
# returns the mean shape, var_matrix, the principal axis and
# a tuple of (variance, percentage of variance)
mean, var_matrix, principal_axis, components = asm.construct()

asm_model = mean, var_matrix, principal_axis, components

# return list of feature vectores from image table
training_data = knn.construct()

# Table like image_table but for the test images.
test_table = []

# get all images
test_list = os.listdir(path_test)

max_count = len(test_list)/2

i = 1

test_images = test_list

# list of all test images and their predictions.
image_results = {}

k = 5

for test_image in test_images:
	if test_image.endswith('.xml'):
		im_struct = parser.init_image(path_test, test_image)
		test_table.append(im_struct)

	# make sure we only test each image one time
	if test_image.endswith('.jpg'):
		print(str(i) + 'of ' + str(max_count))
		# remove the ending of the image
		test_image = test_image[:-4]

		gray_image = parser.get_grayscale(path_test, test_image)
        # get image features

        # TESTER DET ER DUMT
        # gør Otsu agtige ting
        #binary = segmentation.otsu(gray_image)
        # set landmarks
        #landmarks_temp = segmentation.landmark_setter(binary, gray_image)
        #x = aligner.solve_x(mean, landmarks_temp, var_matrix)
        #aligned_landmarks = aligner.align_pair(landmarks_temp, x)
        #image_features = np.dot(principal_axis, aligned_landmarks-mean)
        # NU TESTER VI IKKE MERE

        # return a feature vector + landmarks XXXXXXXXX MÅSKE??? XXXXXX
		image_features, landmarks = asm.image_search(asm_model, gray_image)

        # classify new image from training data
        # get a sorted list of the class id and the number of votes
		label_candidates = knn.classify(training_data, image_features, k)

        # save all label candidates and image_id in dict
		image_results[test_image] = label_candidates

        # label_candidates[0][0] should give the classification
		label = label_candidates[0][0]

		i += 1
        # check if the classification was right and
        # stats.do_shit(test_image, label)

		test_table[-1]['landmarks'] = landmarks
		test_table[-1]['feature_vector'] = image_features

with open('image_dict_labels_with_seg.p', 'wb') as f:
	pickle.dump(image_results, f)
with open('test_table.p', 'wb') as f:
	pickle.dump(test_table, f)
