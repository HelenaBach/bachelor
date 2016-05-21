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

try:
        # get the name of the csv file of the reviews
        path = sys.argv[1] 
except:
        print('The path of the image directory should be passed as argument to this script')
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
        # plot the shape 
 #       xes = landmarks[::2]
 #       yes = landmarks[1::2]
 #       plt.plot(xes,yes)

# show all the shapes in one graph
#plt.show()


# HUSK AT INDSÆTTE PRINT STATEMENTS NÅR DE STORE TRÆNINGSSÆT KØRES
## (SÅ VI HAR ET OVERBLIK OVER HVOR LANGT DET DER KØRER ER)
# align the dataset
mean, var_matrix = aligner.the_real_aligner()

# plot the mean shape
#xes = mean[::2]
#yes = mean[1::2]
#plt.plot(xes,yes)


data = []
for im_struct in image_table:
        landmarks = im_struct['landmarks']
  #      xes = landmarks[::2]
  #      yes = landmarks[1::2]
  #      plt.plot(xes,yes)
        data.append(landmarks)


# plot aligned shapes
#plt.show()

#with open(im_table, 'wb') as f:
#    pickle.dump(image_table, f)
with open('mean_1200.p', 'wb') as f:
    pickle.dump(mean, f)
with open('data_1200.p', 'wb') as f:
    pickle.dump(data, f)
with open('var_matrix_1200.p', 'wb') as f:
    pickle.dump(var_matrix, f) 