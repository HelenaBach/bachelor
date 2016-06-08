import os
import sys
import cv2
import segmentation
import matplotlib.pyplot as plt
import parser
from table import image_table
import aligner
import pca
import asm_uncentered
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

# ROC have fields True Positive (tp), False Positive (fp), number of species (number/p)
# From this, we can find False Negative (fn) = p-tp, and True Negative (tn) = (p+n) - (tp + fp + fn), n = all images 
def update_ROC(class_id, tp=False, fp=False, number=False):
    if class_id in ROC:
        if tp:
            ROC[class_id]['tp'] += 1
        if fp:
            ROC[class_id]['fp'] += 1
        if number:
            ROC[class_id]['number'] += 1
    else:
        ROC[class_id] = {}
        if tp:
            ROC[class_id]['tp'] = 1
        else:
            ROC[class_id]['tp'] = 0
        if fp:
            ROC[class_id]['fp'] = 1
        else:
            ROC[class_id]['fp'] = 0
        if number:
            ROC[class_id]['number'] = 1
        else:
            ROC[class_id]['number'] = 0


# update feature vectors in image table
# returns the mean shape, var_matrix, the principal axis and
# a tuple of (variance, percentage of variance)
mean, var_matrix, principal_axis, components = asm_uncentered.construct()

asm_model = (mean, var_matrix, principal_axis, components)

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
ROC = {}

k = 5
correct = 0

for test_image in test_images:
    # make sure we only test each image one time
    if test_image.endswith('.xml'):
        im_struct = parser.init_image(path_test, test_image)
        test_table.append(im_struct)

        print(str(i) + ' of ' + str(max_count))

        # remove the ending of the image
        test_image = test_image[:-4]

        print('image: ', test_image)
        gray_image = parser.get_grayscale(path_test, test_image)
        #test_image = cv2.imread(path_test + test_image + '.jpg', 0)
        # get image features

        # TESTER DET ER DUMT
        # gør Otsu agtige ting
        #binary = segmentation.otsu(gray_image)
        ## set landmarks
        #landmarks = segmentation.landmark_setter(binary, gray_image)
        #x = aligner.solve_x(mean, landmarks, var_matrix)
        #aligned_landmarks = aligner.align_pair(landmarks, x)
        #image_features = np.dot(principal_axis, aligned_landmarks-mean)
        # NU TESTER VI IKKE MERE
        # return a feature vector + landmarks XXXXXXXXX MÅSKE??? XXXXXX
        image_features, landmarks = asm_uncentered.image_search(asm_model, gray_image)


        # classify new image from training data
        # get a sorted list of the class id and the number of votes
        label_candidates = knn.classify(training_data, image_features, k)

        im_struct['landmarks'] = landmarks
        im_struct['feature_vector'] = image_features
        im_struct['label_candidates'] = label_candidates
        im_struct['prediction'] = True
        # label_candidates[0][0] should give the classification
        label = str(int(label_candidates[0][0]))

        class_id = im_struct['class_id']
        print(label, class_id)
        if class_id == label:
            correct += 1
            update_ROC(class_id, tp=True, number=True)
        else:
            update_ROC(class_id, number=True)
            update_ROC(label, fp=True)
            im_struct['prediction'] = False

        i += 1

        #break
print('accuracy: ', correct/ max_count)
print('correct: ', correct)
#with open('image_dict_labels_with_seg.p', 'wb') as f:
#	pickle.dump(image_results, f)
with open('test_table_image_search_hack.p', 'wb') as f:
    pickle.dump(test_table, f)
with open('ROC_table_image_search_hack.p', 'wb') as f:
    pickle.dump(ROC, f)