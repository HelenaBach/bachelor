import os
import sys
import cv2
import segmentation
import matplotlib.pyplot as plt
import parser
from table import image_table
import aligner
import pca
import math
import pickle 

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
images = os.listdir(path)[:450]

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
mean, principal_axis, components = asm.construct()

asm_model = mean, principal_axis, components

# return list of feature vectores from image table
training_data = knn.construct()



# get all images
test_images = os.listdir(test_path)[:450]

# initialize the image table 
for test_image in test_images:
    # make sure we only test each image one time
    if test_image.endswith('.jpg'):
        # remove the ending of the image
        test_image = test_image[:-4]
        
        gray_image = parser.get_grayscale(test_path, image)
        # get image features
        # return a feature vector
        image_features = asm.image_search(asm_model, gray_image)

        # classify new image from training data 
        label = knn.classify(training_data, image_features)

        # check if the classification was right and
        stats.do_shit(image, label)



