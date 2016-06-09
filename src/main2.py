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

try:
    seg = sys.argv[2]
except:
    print("Say 'otsu' or 'ims' to specify which segmentation to use. ")
    sys.exit(2)


# list of all test images and their predictions.
ROC = {}

# ROC have fields True Positive (tp), False Positive (fp), number of species (number/p)
# From this, we can find False Negative (fn) = p-tp, and True Negative (tn) = (p+n) - (tp + fp + fn), n = all images
def update_ROC(ROC, class_id, tp=False, fp=False, number=False):
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

def create_tables():

    # Table like image_table but for the test images.
    test_table = []

    # get all images
    test_list = os.listdir(path_test)

    pes = [13, 28, 50, 80]

    for p in pes:
        test_table = []
        # update feature vectors in image table
        # returns the mean shape, var_matrix, the principal axis and
        # a tuple of (variance, percentage of variance)
        mean, var_matrix, principal_axis, components = asm.construct(p)

        asm_model = (mean, var_matrix, principal_axis, components)

        max_count = len(test_list)/2

        i = 1

        for test_image in test_list:
            # make sure we only test each image one time
            if test_image.endswith('.xml'):
                im_struct = parser.init_image(path_test, test_image)

                print(str(i) + ' of ' + str(max_count))

                # remove the ending of the image
                test_image = test_image[:-4]

                gray_image = parser.get_grayscale(path_test, test_image)

                if seg == 'otsu':
                    # OTSU
                    binary = segmentation.otsu(gray_image)
                    ## set landmarks
                    landmarks = segmentation.landmark_setter(binary, gray_image)
                    x = aligner.solve_x(mean, landmarks, var_matrix)
                    aligned_landmarks = aligner.align_pair(landmarks, x)
                    # find the image features
                    image_features = np.dot(principal_axis, aligned_landmarks-mean)
                else:
                    # IMAGE SEARCH
                    image_features, landmarks = asm_uncentered.image_search(asm_model, gray_image)

                im_struct['landmarks'] = landmarks
                im_struct['feature_vector'] = image_features

                i += 1

                test_table.append(im_struct)

        with open('p_files/test_table_' + seg + '_pc' + str(p) + '.p', 'wb') as f:
            pickle.dump(test_table, f)


def classify(p):

    # return list of feature vectores from image table
    training_data = knn.construct(p)

    kes = [3, 5, 7, 9]

    with open('p_files/test_table_' + seg + '_pc' + str(p) + '.p', 'rb') as f:
        test_table = pickle.load(f)

    max_count = len(test_table)

    for k in kes:
        print('seg: ' + seg + ' - k : ' + str(k))
        i = 1
        correct = 0
        ROC = {}
        for im_struct in test_table:

            # classify new image from training data
            # get a sorted list of the class id and the number of votes
            label_candidates = knn.classify(training_data, im_struct['feature_vector'], k)

            # label_candidates[0][0] should give the classification
            label = str(int(label_candidates[0][0]))

            im_struct['label_candidates'] = label_candidates
            im_struct['prediction'] = True
            class_id = im_struct['class_id']

            print(label, class_id)
            if class_id == label:
                correct += 1
                ROC = update_ROC(ROC, class_id, tp=True, number=True)
            else:
                ROC = update_ROC(ROC, class_id, number=True)
                ROC = update_ROC(ROC, label, fp=True)
                im_struct['prediction'] = False

            i += 1

        print('accuracy: ', correct/ max_count)
        print('correct: ', correct)

        with open('p_files/test_table_' + seg + '_pc' + str(p) + '_k' + str(k) + '.p', 'wb') as f:
                pickle.dump(test_table, f)

        with open('p_files/ROC_table_' + seg + '_pc' + str(p) + '_k' + str(k) + '.p', 'wb') as f:
            pickle.dump(ROC, f)


create_tables()
#classify(13) #-> p = [13, 28, 50, 80]
