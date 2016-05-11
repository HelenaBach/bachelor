import cv2
import numpy as np
#import math
import sys # debugging
from matplotlib import pyplot as plt
from table import image_table
import aligner
import parser 


def construct():
	with open('mean.p', 'rb') as f:
	    mean = pickle.load(f)
	
	with open('data.p', 'rb') as f:
	    data = pickle.load(f)
	
	#mean = aligner.the_real_aligner()
	#
	#data = []
	#for im_struct in image_table:
	#        landmarks = im_struct['landmarks']
	#        data.append(landmarks)

	
	principal_axis, comp_variance = pca.fit(data, mean, 0.90)

	for im_struct in image_table:
		# mean centred shape
		shape = im_struct['landmarks'] - mean

		# translate data into PCA space
		im_struct['feature_vector'] = np.dot(principal_axis, shape)

	return mean, principal_axis, comp_variance


def image_search(asm_model, image):
	
	#laplacian = cv2.Laplacian(image,cv2.CV_64F) 
	sobelx = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=5)
	sobely = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=5)
	
	#cv2.imshow('image', sobelx)
	#cv2.waitKey(0)
	#cv2.imshow('image', sobely)
	#cv2.waitKey(0)
	#cv2.imshow('image', image)
	#cv2.waitKey(0)
	#cv2.imshow('image', sobely)
	#cv2.waitKey(0)
	#image2 = sobelx**2 + sobely**2
	
	image_diff = abs(sobelx) + abs(sobely)
	#plt.subplot(2,2,1),plt.imshow(image,cmap = 'gray')
	#plt.title('Original'), plt.xticks([]), plt.yticks([])
	#plt.subplot(2,2,2),plt.imshow(image_diff,cmap = 'gray')
	#plt.title('Differentiated image'), plt.xticks([]), plt.yticks([])
	#plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
	#plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
	#plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
	#plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
	#
	#plt.show()
	mean, principal_axis, comp_variance = asm_model

	adjustments_points_normals = []

	xes = mean[::2]
	yes = mean[1::2]

	for i in range(len(xes)):
		x = xes[i]
		y = yes[i]
		x_left = xes[i-1]
		y_left = yes[i-1]
		# wrap around
		x_right = xes[i+1 % len(xes)]
		y_right = yes[i+1 % len(yes)]

		# if something is weird - LOOK HERE!
		line_left  = np.array((x-x_left, y-y_left))
		line_right = np.array((x_right-x, y_right-y))

		norm_left  = get_norm(line_left) 
		norm_right = get_norm(line_right)
		# norm_left and  norm_right is unit length
		norm = (norm_left + norm_right) / 2
		adjustments_points_normals.append(x, y, norm)



def get_norm(coordinats):
	x = coordinats[0]
	y = coordinats[1]
	# fordi det med Louises arme hvor albuen er origo
	# normalize the vectors
	length_of_vector = math.sqrt((x^*2+y^*2)
	return np.array((-y/length_of_vector, x/length_of_vector))

	

