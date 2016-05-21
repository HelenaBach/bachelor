import cv2
import numpy as np
#import math
import sys # debugging
from matplotlib import pyplot as plt
from table import image_table
import aligner
import parser
import pickle
import pca



def construct():
	with open('mean.p', 'rb') as f:
	    mean = pickle.load(f)

	with open('image_table.p', 'rb') as f:
	    data = pickle.load(f)

	with open('var_matrix.p', 'rb') as f:
	    var_matrix = pickle.load(f)

        # If the data is not pre-aligned, aligner.the_real_aligner() should
        # be run.

	data = []
	for im_struct in image_table:
	        landmarks = im_struct['landmarks']
	        data.append(landmarks)


	principal_axis, comp_variance = pca.fit(data, mean, 0.95)

	#print('95')
#	#print(len(principal_axis))
#
	#principal_axis, comp_variance = pca.fit(data, mean, 0.975)
	#print('97.5')
	#print(len(principal_axis))

	for im_struct in image_table:
		# mean centred shape
		shape = im_struct['landmarks'] - mean

		# translate data into PCA space
		im_struct['feature_vector'] = np.dot(principal_axis, shape)

	return mean, var_matrix, principal_axis, comp_variance

def image_search():
	# asm_model, image):
	image = cv2.imread('../data/leafscan/27.jpg', 0)
	sobelx = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=5)
	sobely = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=5)
	#laplacian = cv2.Laplacian(image,cv2.CV_64F)

	#cv2.imshow('image', sobelx)
	#cv2.waitKey(0)
	#cv2.imshow('image', sobely)
	#cv2.waitKey(0)
	#cv2.imshow('image', image)
	#cv2.waitKey(0)
	#cv2.imshow('image', sobely)
	#cv2.waitKey(0)
	#image2 = sobelx**2 + sobely**2

	#image_diff = abs(sobelx) + abs(sobely)

	#plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
	#plt.subplot(2,2,1),plt.imshow(laplacian,cmap = 'gray')
	#plt.subplot(2,2,2),plt.imshow(image_diff,cmap = 'gray')
	#plt.title('Differentiated image'), plt.xticks([]), plt.yticks([])
	#plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
	#plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
	#plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
	#plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
	#
	#plt.show()

	mean, var_matrix, principal_axis, comp_variance = asm_model

	adjustments_points_normals = []
	dX = []
	# diff the image
	image_diff = abs(sobelx) + abs(sobely)

	# Should this be the landmarks?
	temp_landmarks = mean # med noget init placering
	temp_alignment_parameters = np.array((0,0,0,0)) # initial parameters should be something else than 0

	# converges loop
	xes = temp_landmarks[::2]
	yes = temp_landmarks[1::2]

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
		# norm_left and  norm_right are unit length
		norm = (norm_left + norm_right) / 2

		norm_list = []
		for i in range(-2, 3):
			x_diff = nearest_pixel(x + i*norm[0])
			y_diff = nearest_pixel(y + i*norm[1])

			# x, y or y,x ?
			norm_list.append((x_diff, y_diff, image_diff[x_diff][y_diff]))

		# should maybe be min?
		(x_diff, y_diff, pix_value) = max(norm_list,key=lambda item:item[2])

		dX.append(x-x_diff)
		dX.append(y-y_diff)

	# align X o be as close to the new points as possible
	# alignment_parameters = a_x, a_y, t_x, t_y
	diff_alignment_parameters = aligner.solve_x(X+dX, X, var_matrix)
	# HVORDAN GÅR VI FRA A_X, A_Y TIL THETA, SKALERING OG BLA BLA









def get_norm(coordinats):
	x = coordinats[0]
	y = coordinats[1]
	# fordi det med Louises arme hvor albuen er origo
	# normalize the vectors
	length_of_vector = math.sqrt(x**2+y**2)
	return np.array((-y/length_of_vector, x/length_of_vector))
