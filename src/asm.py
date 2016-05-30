import cv2
import numpy as np
#import math
import sys # debugging
from matplotlib import pyplot as plt
from table import image_table
import math # squareroot
import aligner
import parser
import pickle
import pca



def construct():
        with open('mean_more.p', 'rb') as f:
                mean = pickle.load(f)

        with open('image_table_more.p', 'rb') as f:
                old_image_table = pickle.load(f)

        with open('var_matrix_more.p', 'rb') as f:
                var_matrix = pickle.load(f)

        # If the data is not pre-aligned, aligner.the_real_aligner() should
        # be run.

        for img in old_image_table:
                image_table.append(img)

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

def image_search(asm_model, image):

	mean, var_matrix, principal_axis, comp_variance = asm_model

	#image = cv2.imread('../data/leafscan/27.jpg', 0)
	
	# mild smoothing of the image to reduce noise 
	gaussian(image, sigma=0.4)

	sobelx = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=5)
	sobely = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=5)

	# diff the image
	image_diff = np.round(sobelx**2 + sobely**2)

	# initialise the dX array
	diff_image_x = [] # NUMPY ARRAY ISTEDET FOR!!! XXX

	# The landmarks within the model
	model_x = mean # med noget init placering
	# the landmarks within the image
	image_x = model_x
	# initial parameters. s, theta, t_x, t_y
	alignment_parameters = np.array((1,0,0,0))
	# initial b vector
	b = np.array((0))
	b = np.tile(b, len(principal_axis))

	old_suggested_image = image_x

	# converges loop
	for i in range(20): # while True 

	# HUSK AT SIKRER AT ALTING BLIVER GEMT HELE TIDEN - ALLZ THE TIME - WHATEVER U DO!

		diff_image_x = adjustments_along_normals(image_x)
		
		# Test if we are trying to move to the same place as last time
		suggested_image = image_x+diff_image_x
		if suggested_image is old_suggested_image:
			break
		old_suggested_image = suggested_image

		# align X o be as close to the new points as possible
		# alignment_parameters = a_x, a_y, t_x, t_y
		diff_alignment_parameters = aligner.solve_x(image_x+diff_image_x, image_x, var_matrix)

		diff_s, diff_theta, diff_t_x, diff_t_y = get_skale_rotation_translation(diff_alignment_parameters)


		# calculate new s, theta, t_x, t_y
		alignment_parameters = update_parameters(alignment_parameters, diff_s, diff_theta, diff_t_x, diff_t_y)

		# create X_c matrix after updating parameters X_c = X_c + dX_c
		length = int(len(X)/2)
		image_x_c =  get_translation(alignment_parameters[2], alignment_parameters[3], length)

		# y from eq. 19
		y = image_x + diff_image_x - image_x_c

		# suggested movements of the points in the model space
		# dx = M((s(1+ds))^-1, -(theta + dtheta)) [y] - x
		diff_model_x = skale_and_rotate(y, math.pow(alignment_parameters[0], -1), -alignment_parameters[1]) - model_x

		#apply the shape contraints and approximate new model parameter x + dx

		# x + dx ~ mean + P*(b+db) <- allowable shape
		# db = P^t * dx
		# x + dx ~ mean + P*(b+P^t * dx)
		# new b = b+db = b + P^t * dx

		b = b + np.dot(principal_axis, diff_model_x)
		model_x = mean + np.dot(principal_axis, b)

		image_x = skale_and_rotate(model_x, alignment_parameters[0], alignment_parameters[1]) + image_x_c

	return np.dot(principal_axis, model_x)

def adjustments_along_normal(image_x):
	diff_image_x = np.array((0))
	diff_image_x = np.tile(diff_image_x, len(image_x))

	xes = image_x[::2]
	yes = image_x[1::2]

	# find dX
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
		for i in range(-10, 11):
			# round to nearest pixel coordinates
			diff_x = int(round(x + i*norm[0]))
			diff_y = int(round(y + i*norm[1]))

			# x, y or y,x ?
			norm_list.append((diff_x, diff_y, image_diff[diff_x][diff_y]))

		# choose the point with the highest value.
		#(diff_x, diff_y, pix_value) = max(norm_list,key=lambda item:item[2])
		sorted_norms = sorted(norm_list,key=lambda item:item[2], reverse=True)

		best_guess = sorted_norms[0]
		j = 1
		while best_guess[2] == sorted_norms[j][2]:
			best_diff = (x-best_guess[0])**2 + (y-best_guess[1])**2
			new_diff = (x-sorted_norms[j][0])**2 + (y-sorted_norms[j][1])**2
			if new_diff < best_diff:
				best_guess = sorted_norms[j]
			j += 1


		diff_x, diff_y, pix_value = best_guess

		diff_image_x[i*2]   = (x-diff_x)
		diff_image_x[i*2+1] = (y-diff_y)

	return diff_image_x


def get_norm(coordinats):
	x = coordinats[0]
	y = coordinats[1]
	# fordi det med Louises arme hvor albuen er origo
	# normalize the vectors
	length_of_vector = math.sqrt(x**2+y**2)
	return np.array((-y/length_of_vector, x/length_of_vector))


def get_skale_rotation_translation(alignment_parameters):
	a_x = alignment_parameters[0] # s cos theta
	a_y = alignment_parameters[1] # s sin theta
	t_x = alignment_parameters[2]
	t_y = alignment_parameters[3]

	# obtain the suggested skale and rotation:
	# a_x = s * cos(theta), a_y = s * sin(theta)
	# cos(theta)^2 + sin(theta)^2 = 1 (unit circle and pythagoras)
	# s will be the length of a_x + a_y.
	ds = math.sqrt(a_x**2 + a_y**2)
	dtheta =  math.degrees(math.acos(a_x/ds))
	return (ds, dtheta, t_x, t_y)


def get_translation(t_x, t_y, length):

    t = np.array([t_x,t_y])
    # create vector t (translation) of same length as the shape
    # with t_x, t_y repeated
    t = np.tile(t, length)
    return t

def update_parameters(alignment_parameters, diff_s, diff_theta, diff_t_x, diff_t_y):
	s     = alignment_parameters[0]
	theta = alignment_parameters[1]
	t_x   = alignment_parameters[2]
	t_y   = alignment_parameters[3]

	alignment_parameters[0] = s * (1 + diff_s)
	alignment_parameters[1] = theta + diff_theta
	alignment_parameters[2] = t_x + diff_t_x
	alignment_parameters[3] = t_y + diff_t_y

	return alignment_parameters

def skale_and_rotate(shape, s, theta):
    a_x = s * math.cos(theta)
    a_y = s * math.sin(theta)

    M = np.copy(shape)
    #print(M)
    # rotate and scale shape2 (now know as M)
    for k in range(0,n):
        M[k*2]   = (a_x * M[k*2]) - (a_y * M[k*2+1])
        M[k*2+1] = (a_y * M[k*2]) + (a_x * M[k*2+1])

    return M


## HOW TO PLOT ALL THE PICTURES

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