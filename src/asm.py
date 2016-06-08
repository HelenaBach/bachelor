import cv2
import numpy as np
#import math
import sys # debugging
from matplotlib import pyplot as plt
from skimage.filters import gaussian
from table import image_table
import math # squareroot
import aligner
import parser
import pickle
import pca
from scipy.ndimage.filters import gaussian_gradient_magnitude

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

#def image_search(asm_model, image):
#    print('dimensions of image_not_diff:', len(image), ', ', len(image[0]))
#    # image[0] = x
#    # image = y
#    mean, var_matrix, principal_axis, comp_variance = asm_model
#
#    if max_width > image[0] or max(mean_xes) >
    # The landmarks within the model
    model_x = np.copy(mean ) # med noget init placering
    # the landmarks within the image
    image_x = np.copy(mean )
    # initialise the dX array

    #with open('15252_landmarks.p', 'rb') as f:
    #    landmarks_15252 = pickle.load(f)
    #model_x = np.copy(landmarks_15252)
    #image_x = np.copy(landmarks_15252)

    img_xes = image_x[::2]
    img_yes = image_x[1::2]
    plt.plot(img_xes, img_yes)
    plt.show()
=======

    # The landmarks within the model
    model_x = np.copy(mean ) # med noget init placering
>>>>>>> 3c80eca517505bc1888fce25ca36cc7fe96d827e

    diff_image_x = np.array((0.0))
    diff_image_x = np.tile(diff_image_x, len(image_x))

    # initial parameters. s, theta, t_x, t_y
    len_y = np.size(diff_image_x)/2
    len_x = np.size(diff_image_x[0])/2
<<<<<<< HEAD
    alignment_parameters = np.array((0.0, 0.0, 0.0, 0.0)) #float(len_x), float(len_y) ))
=======
    alignment_parameters = np.array((1.0, 0.0, float(len_x), float(len_y) ))
>>>>>>> 3c80eca517505bc1888fce25ca36cc7fe96d827e
    # initial b vector
    b = np.array((0.0))
    b = np.tile(b, len(principal_axis))

    # initial transformation of landmark into the image frame
    translation = get_translation(alignment_parameters[2], alignment_parameters[3], len(image_x)/2)
    image_x = scale_and_rotate(model_x, alignment_parameters[0], alignment_parameters[1]) + translation 
    
    # plot the image
    img_xes = image_x[::2]
    img_yes = image_x[1::2]
    plt.imshow(image_print,cmap = 'gray')
    plt.plot(img_xes, img_yes, marker='.')
    plt.suptitle('Image x ', fontsize = 14)
    plt.show()

    # used to check for significant changes in the suggested movement
    # in image space
    old_suggested_image = image_x

    # converges loop
<<<<<<< HEAD
    for i in range(20): # while True

        # place all point in the image                  x                 y
        #image_x = align_to_image_frame(image_x, len(image_diff[0]), len(image_diff))
=======
    for i in range(4): # while True
>>>>>>> 3c80eca517505bc1888fce25ca36cc7fe96d827e

        # find dX 
        diff_image_x = adjustments_along_normal(image_x, image_diff)
        
        # Test if we are trying to move to the same place as last time
        suggested_image = image_x + diff_image_x
<<<<<<< HEAD
		suc_xes = suggested_image[::2]
		suc_yes = suggested_image[1::2]
		plt.plot(suc_xes, suc_yes)
		plt.show()
=======
        sug_xes = suggested_image[::2] 
        sug_yes = suggested_image[1::2]
        plt.imshow(image_print,cmap = 'gray')
        plt.plot(sug_xes, sug_yes, marker='.')
        plt.suptitle('suggested shape', fontsize = 14)
        plt.show()
>>>>>>> 3c80eca517505bc1888fce25ca36cc7fe96d827e
        #print(suggested_image)
        if (suggested_image is old_suggested_image) or i == 99:
            print('image_search iteration: ' + str(i))
            break
        
        # BRUG DET DER EFTER DEN ER I ALLOWABLE SPACE!!
        old_suggested_image = suggested_image

        # align X o be as close to the new points as possible
        # alignment_parameters = a_x, a_y, t_x, t_y
        diff_alignment_parameters = aligner.solve_x(image_x+diff_image_x, image_x, var_matrix)
        diff_s, diff_theta, diff_t_x, diff_t_y = get_skale_rotation_translation(diff_alignment_parameters)

        length = int(len(image_x)/2)
        image_x_c_old =  get_translation(alignment_parameters[2], alignment_parameters[3], length)
        diff_image_x_c_old =  get_translation(diff_t_x, diff_t_y, length)

        # eq y
        #y = skale_and_rotate(model_x, alignment_parameters[0], alignment_parameters[1]) + diff_image_x - (image_x_c_old + diff_image_x_c_old)

        # calculate new s, theta, t_x, t_y
        alignment_parameters = update_parameters(alignment_parameters, diff_s, diff_theta, diff_t_x, diff_t_y)

        # create X_c matrix after updating parameters X_c = X_c + dX_c
        length = int(len(image_x)/2)
        image_x_c =  get_translation(alignment_parameters[2], alignment_parameters[3], length)

        # y from eq. 19
        y = image_x + diff_image_x - image_x_c

        # suggested movements of the points in the model space
        # dx = M((s(1+ds))^-1, -(theta + dtheta)) [y] - x
        inverted_theta = alignment_parameters[1] * -1
        inverted_s = math.pow(alignment_parameters[0], -1)

        diff_model_x = skale_and_rotate(y, inverted_s, inverted_theta) - model_x

        #apply the shape contraints and approximate new model parameter x + dx

        # 0: x + dx ~ mean + P*(b+db) <- allowable shape
        # 1: db = P^t * dx
        # 2: x + dx ~ mean + P*(b+P^t * dx)
        # 3: new b = b+db = b + P^t * dx
        # obs! PC's is 'transposed' so inverse the transposion

        # update b (3)
        b = b + np.dot(principal_axis, diff_model_x)

        # limit b to be 3 standard deviations from the mean (eq 15)
        for k in range(len(b)):
            if b[k] > 3 * math.sqrt(comp_variance[k][0]):
                print('b was bigger than allowable shape domain')
                b[k] = 3 * math.sqrt(comp_variance[k][0])

            if b[k] < (-3) * math.sqrt(comp_variance[k][0]):
                print('b was smaller than allowable shape domain')
                b[k] = (-3) * math.sqrt(comp_variance[k][0])



        # b coordinats in the model space
        pca_x = np.dot(np.array(principal_axis).transpose(), b)
#        pca_x = np.dot(b, principal_axis)
        # approximate x (0)
        model_x = mean + pca_x



        mod_xes = model_x[::2]
        mod_yes = model_x[1::2]
        plt.imshow(image_print,cmap = 'gray')
        plt.plot(mod_xes, mod_yes, marker='.')
        plt.suptitle('Model shape', fontsize = 14)
        plt.show()


        plt.imshow(image_print,cmap = 'gray')
        img_xes = image_x[::2]
        img_yes = image_x[1::2]
        plt.plot(img_xes, img_yes, marker='o')
        print(image_x_c)

        print(alignment_parameters[0], alignment_parameters[1])
        image_x = skale_and_rotate(model_x, alignment_parameters[0], alignment_parameters[1])# + image_x_c


        img_xes = image_x[::2]
        img_yes = image_x[1::2]
        plt.plot(img_xes, img_yes, marker='.')
        #axes = plt.gca()
        #axes.set_xlim([xmin,xmax])
        #axes.set_ylim([ymin,ymax]
        plt.suptitle('image shape before and after pca', fontsize = 14)
        plt.show()

    # b = P^T(x-mean)
    feature_vector = np.dot(principal_axis, (model_x-mean))

    return feature_vector, model_x #VI ER ENIGE OM AT MODEL_X ER LANDMARKS?? XXXXXXXX

def adjustments_along_normal(image_x, image_diff):
    # image_diff[0] = x
    # image_diff = y
    diff_image_x = np.array((0.0))
    diff_image_x = np.tile(diff_image_x, len(image_x))

    xes = image_x[::2]
    yes = image_x[1::2]

    # find dX
    for i in range(len(xes)):
        print('point nr. ', i)
        x = xes[i]
        y = yes[i]

        x_left = xes[i-1]
        y_left = yes[i-1]
        # wrap around - remember length 100 -> index 0-99
        x_right = xes[(i+1) % len(xes)]
        y_right = yes[(i+1) % len(yes)]

        # if something is weird - LOOK HERE!
        if x-x_left == 0 and y-y_left == 0:
            print('line left is zero. x: ', x, ' y: ', y)
            print(image_x)

        if x_right-x == 0 and y_right-y == 0:
            print('line right is zero. x: ', x, ' y: ', y)
            print(x_right, ' , ', y_right)
            print(image_x)
        line_left  = np.array((x-x_left, y-y_left))
        line_right = np.array((x_right-x, y_right-y))

        norm_left  = get_norm(line_left)
        norm_right = get_norm(line_right)
        # norm_left and  norm_right are unit length
        norm = (norm_left + norm_right) / 2

        norm_list = []
<<<<<<< HEAD
        print('index: x= ', x, ' y= ', y)
        for j in range(-50, 51):
            # round to nearest pixel coordinates
            diff_x = x + j*norm[0]
            diff_y = y + j*norm[1]

            # 0 indx?
            # image_diff[0] = x
            # image_diff = y
=======
        #print('index: x= ', x, ' y= ', y)
        own_diff_value = image_diff[int(round(y))][int(round(x))]
        diff_x_best = 0
        diff_y_best = 0

        diff_x_best, diff_y_best =  x, y
        for j in range(1, 100):

         # round to nearest pixel coordinates
            diff_x = x + j*norm[0]
            diff_y = y + j*norm[1]

         #diff_x_neg = x - j*norm[0]
                  #diff_y_neg = y - j*norm[1]

         # 0 indx? 
        # image_diff[0] = x
                  # image_diff = y
            if diff_x > len(image_diff[0])-1 or diff_x < 0 or \
                diff_y > len(image_diff)-1    or diff_y < 0:
                diff_value = -1
            else:
                diff_value = image_diff[int(round(diff_y))][int(round(diff_x))]

            if diff_value > 1000000 and  diff_value > own_diff_value:
                if image_diff[int(round(diff_y-1))][int(round(diff_x-1))] < diff_value and image_diff[int(round(diff_y+1))][int(round(diff_x+1))] < diff_value:                  
                    print('index: x= ', x, ' y= ', y)
                    print('old diff value: ',   own_diff_value)
                    print('new index: x= ', diff_x, ' y= ', diff_y)
                    print('new diff value: ',   diff_value)
                    diff_x_best, diff_y_best, pix_value_best = diff_x, diff_y, diff_value
                    break                                

        for j in range(100):

         # round to nearest pixel coordinates
            diff_x = x - j*norm[0]
            diff_y = y - j*norm[1]

         #diff_x_neg = x - j*norm[0]
                  #diff_y_neg = y - j*norm[0]

         # 1 indx? 
        # image_diff[0] = x
                  # image_diff = y
>>>>>>> 3c80eca517505bc1888fce25ca36cc7fe96d827e
            if diff_x > len(image_diff[0])-1 or diff_x < 0 or \
                diff_y > len(image_diff)-1    or diff_y < 0:
                diff_value = -1
            else:
                diff_value = image_diff[int(round(diff_y))][int(round(diff_x))]

<<<<<<< HEAD
            norm_list.append((diff_x, diff_y, diff_value))

        #print('norm_list:', norm_list)
        #print(image_diff)
        #for i in range(len(image_diff)):
        #	for j in range(len(image_diff[0])):
        #		if (image_diff[i][j] != 0.0):
        #			print('DIFFIE: ', image_diff[i][j])
        # choose the point with the highest value.
        #(diff_x, diff_y, pix_value) = max(norm_list,key=lambda item:item[2])
        sorted_norms = sorted(norm_list,key=lambda item:item[2], reverse=True)

#        if x > len(image_diff[0])-1 or x < 0 or\
#           y > len(image_diff)-1    or y < 0:
#            print('sorted norms:' , sorted_norms)
        best_guess = sorted_norms[0]
        #sorted_norms[1] = (127, 33, 58888.0)
        #sorted_norms[2] = (128, 30, 58888.0)
        #print(best_guess)
        j = 1
        while  j < len(sorted_norms) and best_guess[2] == sorted_norms[j][2]:
            best_diff = (x-best_guess[0])**2 + (y-best_guess[1])**2
            new_diff = (x-sorted_norms[j][0])**2 + (y-sorted_norms[j][1])**2
            if new_diff < best_diff:
                best_guess = sorted_norms[j]
            j += 1
        print('best guess: ',best_guess)

        diff_x_best, diff_y_best, pix_value_best = best_guess
=======
            if diff_value > 1000000 and diff_value > own_diff_value:
                if image_diff[int(round(diff_y-1))][int(round(diff_x-1))] < diff_value and image_diff[int(round(diff_y+1))][int(round(diff_x+1))] < diff_value:                  
                    diff_x_best, diff_y_best, pix_value_best = diff_x, diff_y, diff_value
                    break

#          norm_list.append((diff_x, diff_y, diff_value))
#      
#        #print('norm_list:', norm_list)
#        #print(image_diff)
#        #for i in range(len(image_diff)):
#        #	for j in range(len(image_diff[0])):
#        #		if (image_diff[i][j] != 0.0):
#        #			print('DIFFIE: ', image_diff[i][j])
#        # choose the point with the highest value.
#        #(diff_x, diff_y, pix_value) = max(norm_list,key=lambda item:item[2])
#        sorted_norms = sorted(norm_list,key=lambda item:item[2], reverse=True)
#
#    image_diff = gaussian_gradient_magnitude(image, sigma=1)
#
#    #image_diff = np.copy(image)
#    # mild smoothing of the image to reduce noise
#    #gaussian(image, sigma=0.4)
#    #sobelx = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=5)
#    #sobely = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=5)
#    ## diff the image
#    #image_diff = np.round(sobelx**2 + sobely**2)
#
#    plt.imshow(image_diff,cmap = 'gray')
#    plt.show()
#
#
#    # The landmarks within the model
#    model_x = np.copy(mean ) # med noget init placering
#
#    diff_image_x = np.array((0.0))
#    diff_image_x = np.tile(diff_image_x, len(image_x))
#
#    # initial parameters. s, theta, t_x, t_y
#    len_y = np.size(diff_image_x)/2
#    len_x = np.size(diff_image_x[0])/2
#    alignment_parameters = np.array((1.0, 0.0, float(len_x), float(len_y) ))
#    # initial b vector
#    b = np.array((0.0))
#    b = np.tile(b, len(principal_axis))
#
#    # initial transformation of landmark into the image frame
#    translation = get_translation(alignment_parameters[2], alignment_parameters[3], len(image_x)/2)
#    image_x = scale_and_rotate(model_x, alignment_parameters[0], alignment_parameters[1]) + translation 
#    
#    # plot the image
#    img_xes = image_x[::2]
#    img_yes = image_x[1::2]
#    plt.imshow(image_print,cmap = 'gray')
#    plt.plot(img_xes, img_yes, marker='.')
#    plt.suptitle('Image x ', fontsize = 14)
#    plt.show()
#
#    # used to check for significant changes in the suggested movement
#    # in image space
#    old_suggested_image = image_x
#
#    # converges loop
#    for i in range(4): # while True
#
#        # find dX 
#        diff_image_x = adjustments_along_normal(image_x, image_diff)
#        
#        # Test if we are trying to move to the same place as last time
#        suggested_image = image_x + diff_image_x
#        sug_xes = suggested_image[::2] 
#        sug_yes = suggested_image[1::2]
#        plt.imshow(image_print,cmap = 'gray')
#        plt.plot(sug_xes, sug_yes, marker='.')
#        plt.suptitle('suggested shape', fontsize = 14)
#        plt.show()
#        #print(suggested_image)
#        if (suggested_image is old_suggested_image) or i == 99:
#            print('image_search iteration: ' + str(i))
#            break
#        
#        # BRUG DET DER EFTER DEN ER I ALLOWABLE SPACE!!
#        old_suggested_image = suggested_image
#
#        # align X o be as close to the new points as possible
#        # alignment_parameters = a_x, a_y, t_x, t_y
#        diff_alignment_parameters = aligner.solve_x(image_x+diff_image_x, image_x, var_matrix)
#        diff_s, diff_theta, diff_t_x, diff_t_y = get_skale_rotation_translation(diff_alignment_parameters)
#
#        length = int(len(image_x)/2)
#        image_x_c_old =  get_translation(alignment_parameters[2], alignment_parameters[3], length)
#        diff_image_x_c_old =  get_translation(diff_t_x, diff_t_y, length)
#
#        # eq y
#        #y = skale_and_rotate(model_x, alignment_parameters[0], alignment_parameters[1]) + diff_image_x - (image_x_c_old + diff_image_x_c_old)
#
#        # calculate new s, theta, t_x, t_y
#        alignment_parameters = update_parameters(alignment_parameters, diff_s, diff_theta, diff_t_x, diff_t_y)
#
#        # create X_c matrix after updating parameters X_c = X_c + dX_c
#        length = int(len(image_x)/2)
#        image_x_c =  get_translation(alignment_parameters[2], alignment_parameters[3], length)
#
#        # y from eq. 19
#        y = image_x + diff_image_x - image_x_c
#
#        # suggested movements of the points in the model space
#        # dx = M((s(1+ds))^-1, -(theta + dtheta)) [y] - x
#        inverted_theta = alignment_parameters[1] * -1
#        inverted_s = math.pow(alignment_parameters[0], -1)
#        diff_model_x = skale_and_rotate(y, inverted_s, inverted_theta) - model_x
#
#        #apply the shape contraints and approximate new model parameter x + dx
#
#        # 0: x + dx ~ mean + P*(b+db) <- allowable shape
#        # 1: db = P^t * dx
#        # 2: x + dx ~ mean + P*(b+P^t * dx)
#        # 3: new b = b+db = b + P^t * dx
#        # obs! PC's is 'transposed' so inverse the transposion
#
#        # update b (3)
#        b = b + np.dot(principal_axis, diff_model_x)
#        
#        # limit b to be 3 standard deviations from the mean (eq 15)
#        for k in range(len(b)):
#            if b[k] > 3 * math.sqrt(comp_variance[k][0]):
#                print('b was bigger than allowable shape domain')
#                b[k] = 3 * math.sqrt(comp_variance[k][0])
#
#            if b[k] < (-3) * math.sqrt(comp_variance[k][0]):
#                print('b was smaller than allowable shape domain')
#                b[k] = (-3) * math.sqrt(comp_variance[k][0])
#
#
#
#        # b coordinats in the model space
#        pca_x = np.dot(np.array(principal_axis).transpose(), b)
##        pca_x = np.dot(b, principal_axis)
#        # approximate x (0) 
#        model_x = mean + pca_x
#
#
#
#        mod_xes = model_x[::2]
#        mod_yes = model_x[1::2]
#        plt.imshow(image_print,cmap = 'gray')
#        plt.plot(mod_xes, mod_yes, marker='.')
#        plt.suptitle('Model shape', fontsize = 14)
#        plt.show()
#
#
#        plt.imshow(image_print,cmap = 'gray')
#        img_xes = image_x[::2]
#        img_yes = image_x[1::2]
#        plt.plot(img_xes, img_yes, marker='o')
#        print(image_x_c)
#
#        print(alignment_parameters[0], alignment_parameters[1])
#        image_x = skale_and_rotate(model_x, alignment_parameters[0], alignment_parameters[1])# + image_x_c
#
#
#        img_xes = image_x[::2]
#        img_yes = image_x[1::2]
#        plt.plot(img_xes, img_yes, marker='.')
#        #axes = plt.gca()
#        #axes.set_xlim([xmin,xmax])
#        #axes.set_ylim([ymin,ymax]
#        plt.suptitle('image shape before and after pca', fontsize = 14)
#        plt.show()
#
#    # b = P^T(x-mean)
#    feature_vector = np.dot(principal_axis, (model_x-mean))
#
#    return feature_vector, model_x #VI ER ENIGE OM AT MODEL_X ER LANDMARKS?? XXXXXXXX
#
#def adjustments_along_normal(image_x, image_diff):
#    # image_diff[0] = x
#    # image_diff = y
#    diff_image_x = np.array((0.0))
#    diff_image_x = np.tile(diff_image_x, len(image_x))
#
#    xes = image_x[::2]
#    yes = image_x[1::2]
#
#    # find dX
#    for i in range(len(xes)):
#        print('point nr. ', i)
#        x = xes[i]
#        y = yes[i]
#
#        x_left = xes[i-1]
#        y_left = yes[i-1]
#        # wrap around - remember length 100 -> index 0-99
#        x_right = xes[(i+1) % len(xes)]
#        y_right = yes[(i+1) % len(yes)]
#
#        # if something is weird - LOOK HERE!
#        if x-x_left == 0 and y-y_left == 0:
#            print('line left is zero. x: ', x, ' y: ', y)
#            print(image_x)
#
#        if x_right-x == 0 and y_right-y == 0:
#            print('line right is zero. x: ', x, ' y: ', y)
#            print(x_right, ' , ', y_right)
#            print(image_x)
#        line_left  = np.array((x-x_left, y-y_left))
#        line_right = np.array((x_right-x, y_right-y))
#
#        norm_left  = get_norm(line_left)
#        norm_right = get_norm(line_right)
#        # norm_left and  norm_right are unit length
#        norm = (norm_left + norm_right) / 2
#
#        norm_list = []
#        #print('index: x= ', x, ' y= ', y)
#        own_diff_value = image_diff[int(round(y))][int(round(x))]
#        diff_x_best = 0
#        diff_y_best = 0
#
#        diff_x_best, diff_y_best =  x, y
#        for j in range(1, 100):
#
#         # round to nearest pixel coordinates
#            diff_x = x + j*norm[0]
#            diff_y = y + j*norm[1]
#
#         #diff_x_neg = x - j*norm[0]
#                  #diff_y_neg = y - j*norm[1]
#
#         # 0 indx? 
#        # image_diff[0] = x
#                  # image_diff = y
#            if diff_x > len(image_diff[0])-1 or diff_x < 0 or \
#                diff_y > len(image_diff)-1    or diff_y < 0:
#                diff_value = -1
#            else:
#                diff_value = image_diff[int(round(diff_y))][int(round(diff_x))]
#
#            if diff_value > 1000000 and  diff_value > own_diff_value:
#                if image_diff[int(round(diff_y-1))][int(round(diff_x-1))] < diff_value and image_diff[int(round(diff_y+1))][int(round(diff_x+1))] < diff_value:                  
#                    print('index: x= ', x, ' y= ', y)
#                    print('old diff value: ',   own_diff_value)
#                    print('new index: x= ', diff_x, ' y= ', diff_y)
#                    print('new diff value: ',   diff_value)
#                    diff_x_best, diff_y_best, pix_value_best = diff_x, diff_y, diff_value
#                    break                                
#
#        for j in range(100):
#
#         # round to nearest pixel coordinates
#            diff_x = x - j*norm[0]
#            diff_y = y - j*norm[1]
#
#         #diff_x_neg = x - j*norm[0]
#                  #diff_y_neg = y - j*norm[0]
#
#         # 1 indx? 
#        # image_diff[0] = x
#                  # image_diff = y
#            if diff_x > len(image_diff[0])-1 or diff_x < 0 or \
#                diff_y > len(image_diff)-1    or diff_y < 0:
#                diff_value = -1
#            else:
#                diff_value = image_diff[int(round(diff_y))][int(round(diff_x))]
#
#            if diff_value > 1000000 and diff_value > own_diff_value:
#                if image_diff[int(round(diff_y-1))][int(round(diff_x-1))] < diff_value and image_diff[int(round(diff_y+1))][int(round(diff_x+1))] < diff_value:                  
#                    diff_x_best, diff_y_best, pix_value_best = diff_x, diff_y, diff_value
#                    break
#
##          norm_list.append((diff_x, diff_y, diff_value))
##      
##        #print('norm_list:', norm_list)
##        #print(image_diff)
##        #for i in range(len(image_diff)):
##        #	for j in range(len(image_diff[0])):
##        #		if (image_diff[i][j] != 0.0):
##        #			print('DIFFIE: ', image_diff[i][j])
##        # choose the point with the highest value.
##        #(diff_x, diff_y, pix_value) = max(norm_list,key=lambda item:item[2])
##        sorted_norms = sorted(norm_list,key=lambda item:item[2], reverse=True)
##
###        if x > len(image_diff[0])-1 or x < 0 or\
###           y > len(image_diff)-1    or y < 0:
###            print('sorted norms:' , sorted_norms)
##        best_guess = sorted_norms[0]
##        #sorted_norms[1] = (127, 33, 58888.0)
##        #sorted_norms[2] = (128, 30, 58888.0)
##        #print(best_guess) 
##        j = 1
##        while  j < len(sorted_norms) and best_guess[2] == sorted_norms[j][2]:
##            best_diff = (x-best_guess[0])**2 + (y-best_guess[1])**2
##            new_diff = (x-sorted_norms[j][0])**2 + (y-sorted_norms[j][1])**2
##            if new_diff < best_diff:
##                best_guess = sorted_norms[j]
##            j += 1
##        print('best guess: ',best_guess)
##
##        diff_x_best, diff_y_best, pix_value_best = best_guess
#        #print('chosen: diff_x= ', diff_x, ' diff_y= ', diff_y)
#        #print('simple math:', 128.045957-128)
#        diff_image_x[i*2]   = (x-diff_x_best)
#        diff_image_x[i*2+1] = (y-diff_y_best)
#        #print('j:', j)
#        #print('dX x: ', diff_image_x[i*2])
#        #print('dX x: ', diff_image_x[i*2+1])
#        #sys.exit()
#    return diff_image_x
#
#def get_norm(coordinats):
#    x = coordinats[0]
#    y = coordinats[1]
#    # fordi det med Louises arme hvor albuen er origo
#    # normalize the vectors
#    length_of_vector = math.sqrt(x**2+y**2)
#    if length_of_vector == 0:
#        print('length is zero, x: ', x, 'y: ', y)
#    return np.array((-y/length_of_vector, x/length_of_vector))
#
#def get_skale_rotation_translation(alignment_parameters):
#	a_x = alignment_parameters[0] # s cos theta
#	a_y = alignment_parameters[1] # s sin theta
#	t_x = alignment_parameters[2]
#	t_y = alignment_parameters[3]
#
#	# obtain the suggested skale and rotation:
#	# a_x = s * cos(theta), a_y = s * sin(theta)
#	# cos(theta)^2 + sin(theta)^2 = 1 (unit circle and pythagoras)
#	# s will be the length of a_x + a_y.
#	ds = math.sqrt(a_x**2 + a_y**2)
#	dtheta =  math.degrees(math.acos(a_x/ds))
#	return (ds, dtheta, t_x, t_y)
#
#def get_translation(t_x, t_y, length):
#    t = np.array([t_x,t_y])
#    # create vector t (translation) of same length as the shape
#    # with t_x, t_y repeated
#    t = np.tile(t, length)
#    return t
#
#def update_parameters(alignment_parameters, diff_s, diff_theta, diff_t_x, diff_t_y):
#    s     = alignment_parameters[0]
#    theta = alignment_parameters[1]
#    t_x   = alignment_parameters[2]
#    t_y   = alignment_parameters[3]
#
#    alignment_parameters[0] =  float(s * diff_s) #s * (1 + diff_s)
#    alignment_parameters[1] = theta + diff_theta
#    alignment_parameters[2] = t_x + diff_t_x
#    alignment_parameters[3] = t_y + diff_t_y
#
#    return alignment_parameters
#
#def skale_and_rotate(shape, s, theta):
#	a_x = s * math.cos(theta)
#	a_y = s * math.sin(theta)
#
#	M = np.copy(shape)
#    #print(M)
#    # rotate and scale shape (now know as M)
#	for k in range(0,int(len(shape)/2)):
#		M[k*2]   = (a_x * M[k*2]) - (a_y * M[k*2+1])
#		M[k*2+1] = (a_y * M[k*2]) + (a_x * M[k*2+1])
#
#	return M
#
#
#with open('test_image_search_image.p', 'rb') as f:
#	image = pickle.load(f)
#
#with open('test_image_search_model.p', 'rb') as f:
#	asm_model = pickle.load(f)
#
##print('tester image: ', image)
#image_print = cv2.imread('../data/test/15252.jpg', 0)
#plt.imshow(image_print,cmap = 'gray')
#plt.show()
#
#image_search(asm_model, image)
#
#
#
### HOW TO PLOT ALL THE PICTURES
#
##plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
##plt.subplot(2,2,1),plt.imshow(laplacian,cmap = 'gray')
##plt.subplot(2,2,2),plt.imshow(image_diff,cmap = 'gray')
##plt.title('Differentiated image'), plt.xticks([]), plt.yticks([])
##plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
##plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
##plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
##plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
##
#	#plt.show()
#