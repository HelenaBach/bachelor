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
import math
import os

def construct(p):
    with open('p_files/mean.p', 'rb') as f:
        mean = pickle.load(f)

    with open('p_files/image_table.p', 'rb') as f:
        old_image_table = pickle.load(f)

    with open('p_files/var_matrix.p', 'rb') as f:
        var_matrix = pickle.load(f)

    # If the data is not pre-aligned, aligner.the_real_aligner() should
    # be run.

#    # hack to fill the image table
#    for img in old_image_table:
#        image_table.append(img)

    # create data list
    data = []
    for im_struct in old_image_table:
        landmarks = im_struct['landmarks']
        data.append(landmarks)

    # run pca
    principal_axis, comp_variance = pca.fit(data, mean, p)
    print('data is fitted')

    for im_struct in old_image_table:
        # mean centred shape
        shape = im_struct['landmarks'] - mean

        # translate data into PCA space
        im_struct['feature_vector'] = np.dot(principal_axis, shape)

    with open('p_files/image_table_p' + str(p) + '.p', 'wb') as f:
        pickle.dump(old_image_table, f)

    print('image table is dumped')
    return mean, var_matrix, principal_axis, comp_variance


# find landmarks of the image using image search
# arguments: the asm model which is used to find the shape within the image
#          : the image of the shape to be found
#          : the threshold
# returns
def image_search(asm_model, image, threshold=70):
    # image[0] = x
    # image = y
    mean, var_matrix, principal_axis, comp_variance = asm_model

    # differentiate the image
    image_tmp = np.array(image, dtype=np.float)
    sigma = 2
    image_diff = sigma * gaussian_gradient_magnitude(image_tmp, sigma=sigma)
    image_diff = normalize_image(image_diff)

#    plt.imshow(image_diff,cmap = 'gray')
#    plt.show()

    # The landmarks within the model is the meanshape
    model_x = np.copy(mean )

    # normalize model_x to align the leaf in the image as good as possible
    a_x, a_y, lower_x, lower_y = normalize_model(model_x, image_diff, threshold, image)

    # rotate model_x
    rotation_matrix = build_matrix(a_x, a_y)
    model_x_rotated = scale_and_rotate(model_x, rotation_matrix)

    # find initial translation of model_x
    t_x = lower_x - model_x_rotated[0]
    t_y = lower_y - model_x_rotated[1]
    length = len(model_x)/2
    t_vector = get_translation(t_x, t_y, length)

    # initial landmarks within the image
    image_x = model_x_rotated + t_vector
    print('initial t_values:', t_x, t_y)


    #principal_axis = principal_axis[:20]

     # initial b vector
    b = np.array((0.0))
    b = np.tile(b, len(principal_axis))

#    # plot of the initial placement of the landmark within the image frame
#    img_xes = image_x[::2]
#    img_yes = image_x[1::2]
#    plt.imshow(image,cmap = 'gray')
#    plt.plot(img_xes, img_yes, marker='.')
#    plt.suptitle('Initial placement of image x ', fontsize = 14)
#    plt.show()

    # instantiate the approximation of dx
    approx_dx = np.array((0))
    approx_dx = np.tile(approx_dx, len(model_x))

    # this loop should ideally run until the landmarks reach a fix point
    for i in range(50): # while True
        print('iteration ', i)
        # find dX -> the suggested changes in the image frame
        diff_image_x = adjustments_along_normal(image_x, image_diff, threshold)

#        img_xes = image_x[::2]
#        img_yes = image_x[1::2]
#        plt.plot(img_xes, img_yes, marker='.')
#
#        diff_image_xes = (diff_image_x + image_x)[::2]
#        diff_image_yes = (diff_image_x + image_x)[1::2]
#        plt.imshow(image_diff, cmap='gray')
#        plt.plot(diff_image_xes, diff_image_yes, marker='.')
#        plt.suptitle('suggested adjustments')
#        plt.show()

        # align X o be as close to the new points as possible
        # alignment_parameters = a_x, a_y, t_x, t_y

        diff_alignment_parameters = aligner.solve_x(image_x+diff_image_x, image_x, var_matrix)
        #print(diff_alignment_parameters)

        diff_matrix = build_matrix(diff_alignment_parameters[0], diff_alignment_parameters[1])
        length = len(image_x)/2
        #diff_vector = get_translation(diff_alignment_parameters[3], diff_alignment_parameters[2], length)
        diff_vector = get_translation(diff_alignment_parameters[2], diff_alignment_parameters[3], length)
        #print('diff t vector: ', diff_vector[0], diff_vector[1])

        # y from eq. 19
        #y = image_x + diff_image_x - image_x_c
        # +
        y = scale_and_rotate(model_x + approx_dx, rotation_matrix) + diff_image_x - diff_vector

        # calculate new s, theta, t_x, t_y
        #alignment_parameters = np.dot(alignment_parameters, diff_alignment_parameters)
        rotation_matrix = np.dot(rotation_matrix, diff_matrix)
        t_vector = t_vector + diff_vector
        #print('new t_value:', t_vector[0], t_vector[1])
        #print('diff t value: ', diff_vector[0], diff_vector[1])

        # suggested movements of the points in the model space
        # dx = M((s(1+ds))^-1, -(theta + dtheta)) [y] - x
        diff_model_x = scale_and_rotate(y, np.linalg.inv(rotation_matrix)) - model_x

#        # plot the x + dx
#        diff_model_xes = (model_x + diff_model_x)[::2]
#        diff_model_yes = (model_x + diff_model_x)[1::2]
#        plt.plot(diff_model_xes, diff_model_yes, color='purple')
#        plt.show()

        #apply the shape contraints and approximate new model parameter x + dx
        # 0: x + dx ~ x + P*(b+db) <- allowable shape
        # 1: db = P^t * dx
        # 2: x + dx ~ x + P*(b+P^t * dx)
        # 3: b = b + db = b + P^t * dx

        # obs! PC's is 'transposed' so inverse the transposion

        # update b (3)
        db = np.dot(principal_axis, diff_model_x)

        # since b = [0,..,0] for mean, and x is mean -> b + db = db
        b = db

        # limit b to be 3 standard deviations from the mean (eq 15)
        for k in range(len(b)):
            if b[k] > 3 * math.sqrt(comp_variance[k][0]):
                print('b was bigger than allowable shape domain')
                print('b index: ', k)
                b[k] = 3 * math.sqrt(comp_variance[k][0])

            if b[k] < (-3) * math.sqrt(comp_variance[k][0]):
                print('b was smaller than allowable shape domain')
                print('b index: ', k)
                b[k] = (-3) * math.sqrt(comp_variance[k][0])

        # b coordinats in the model space
        approx_dx = np.dot(np.array(principal_axis).transpose(), db)

#        # plot the model
#        mod_xes = (model_x+approx_dx)[::2]
#        mod_yes = (model_x+approx_dx)[1::2]
#        plt.plot(mod_xes, mod_yes, marker='.')
#        plt.suptitle('Model shape', fontsize = 14)
#        plt.show()



        # store the old suggestion of landmark to test if any change has happend
        image_x_old = image_x
 #       #image_x_old = scale_and_rotate(model_x+approx_dx, rotation_matrix)
 #       # plot the landmarks within the image frame
 #       plt.imshow(image_diff,cmap = 'gray')
 #       img_xes = image_x_old[::2]
 #       img_yes = image_x_old[1::2]
 #       plt.plot(img_xes, img_yes, marker='o')

        image_x = scale_and_rotate(model_x+approx_dx, rotation_matrix) + t_vector

         #find initial translation of model_x
        if i == 0:
#            print('lower x og y  : ', lower_x, lower_y)
#            print('image x og y  : ', image_x[0], image_x[1])
#            print('t vector value: ', t_vector[0], t_vector[1])
            t_x_hack = lower_x - image_x[0]
            t_y_hack = lower_y - image_x[1]
#        print('hack vector t : ', t_x_hack, t_y_hack)
            length = len(image_x)/2
            t_vector_hack = get_translation(t_x_hack, t_y_hack, length)
            image_x = image_x + t_vector_hack
#        #print('diff in image frame: ', sum(abs(image_x-image_x_old)))

        # further plot of the landmarks within the image frame.
#        img_xes = image_x[::2]
#        img_yes = image_x[1::2]
#        plt.plot(img_xes, img_yes, marker='.')
#        #axes = plt.gca()
#        #axes.set_xlim([xmin,xmax])
#        #axes.set_ylim([ymin,ymax]
#        plt.suptitle('image shape before and after pca', fontsize = 14)
#        plt.show()
        if sum(abs(image_x-image_x_old)) < 100:
            break
    return b, (model_x + approx_dx)

# find suggested changes in the image frame
# arguments: image_x   : the current coordinats in the image frame,
#            image_diff: the differentiated image
#            threshold : the threshold of when to choose another coordinat than the original
# return   : list of the suggested changes in the image frame
def adjustments_along_normal(image_x, image_diff, threshold):
    # image_diff[0] = x
    # image_diff = y

    # the suggested changes in the image frame
    diff_image_x = np.array((0.0))
    diff_image_x = np.tile(diff_image_x, len(image_x))

    xes = image_x[::2]
    yes = image_x[1::2]

    #
    points_got_same_coordinats = 0
    # find dX -> the suggested changes in the image frame
    for i in range(len(xes)):
        #print('point nr. ', i)
        x = xes[i]
        y = yes[i]

        # the point to the left of current point
        x_left = xes[i-1]
        y_left = yes[i-1]
        #
        # wrap around
        x_right = xes[(i+1) % len(xes)]
        y_right = yes[(i+1) % len(yes)]

        # if the two points are placed at the same coordinat
        if x-x_left == 0 and y-y_left == 0:
            k = 2
            # use the next point
            while x-x_left == 0 and y-y_left == 0:
                x_left = xes[i-k]
                y_left = yes[i-k]
                k += 1
                # if all points have the same coordinates
                if k == len(xes):
                    points_got_same_coordinats = 1
                    print('all points have same coordinates')
                    #sys.exit(2)
                    break
        if points_got_same_coordinats == 1:
            break

        # if the two points are placed at the same coordinat
        if x_right-x == 0 and y_right-y == 0:
            l = 2
            while x-x_left == 0 and y-y_left == 0:
                x_right = xes[(i+l) % len(xes)]
                y_right = yes[(i+l) % len(yes)]
                l += 1
                # if all points have the same coordinates
                # we don't look at the points that x_left has already looked at
                if l == len(xes)-k:
                    print('all points have same coordinates')
                    points_got_same_coordinats = 1
                    #sys.exit(2)
                    break

        if points_got_same_coordinats == 1:
            break

        line_left  = np.array((x-x_left, y-y_left))
        line_right = np.array((x_right-x, y_right-y))

        # norm_left and  norm_right are unit length
        norm_left  = get_norm(line_left)
        norm_right = get_norm(line_right)

        # norm is unit length
        norm = (norm_left + norm_right) / 2


        # initialize the best choise as the original point
        # if original point is not in the image frame, then set value to -1
        if x > len(image_diff[0])-1 or x < 0 or \
           y > len(image_diff)-1    or y < 0:
            own_diff_value = -1
        else:
            own_diff_value = image_diff[int(round(y))][int(round(x))]

        # initialize the best choise as the original point
        diff_x_best, diff_y_best =  x, y

        #if i == 10:
        #    print('original x and y', x, y)
        #    print('own diff value: ', own_diff_value)

        for j in range(1, 150):

         # round to nearest pixel coordinates
            diff_x_pos = x + j*norm[0]
            diff_y_pos = y + j*norm[1]

            diff_x_neg = x - j*norm[0]
            diff_y_neg = y - j*norm[1]


            # image_diff[0] = x
            # image_diff = y
            # make sure that the coordinates are within the image
            if diff_x_pos > len(image_diff[0])-1 or diff_x_pos < 0 or \
               diff_y_pos > len(image_diff)-1    or diff_y_pos < 0:
                diff_value_pos = -1
            else:
                diff_value_pos = image_diff[int(round(diff_y_pos))][int(round(diff_x_pos))]
            # make sure that the coordinates are within the image
            if diff_x_neg > len(image_diff[0])-1 or diff_x_neg < 0 or \
               diff_y_neg > len(image_diff)-1    or diff_y_neg < 0:
                diff_value_neg = -1
            else:
                diff_value_neg = image_diff[int(round(diff_y_neg))][int(round(diff_x_neg))]

         #   print('pos x, y: ', diff_x_pos, diff_y_pos)
         #   print('value   : ', diff_value_pos)
            diff_value = diff_value_pos
            diff_y, diff_x = diff_y_pos, diff_x_pos
            flag = 1
        #    print('neg x, y: ', diff_x_neg, diff_y_neg)
        #    print('value   : ', diff_value_neg)
            if diff_value_neg > diff_value_pos:
                diff_value = diff_value_neg
                diff_y, diff_x = diff_y_neg, diff_x_neg
                flag = -1

            if diff_value > threshold and  diff_value > own_diff_value:
                norm_y = norm[1] * flag
                norm_x = norm[0] * flag
                # ensure next value is within the image frame
                if int(round(diff_x + norm_x)) > len(image_diff[0])-1 or int(round(diff_x + norm_x)) < 0 or \
                   int(round(diff_y + norm_y)) > len(image_diff)-1    or int(round(diff_y + norm_y)) < 0:
                    next_diff_value = -1
                else:
                    next_diff_value = image_diff[int(round(diff_y + norm_y))][int(round(diff_x + norm_x))]

                if next_diff_value < diff_value:
                    diff_x_best, diff_y_best, pix_value_best = diff_x, diff_y, diff_value
                    #print('I CHOOSE YOOU!')
                    break

        diff_image_x[i*2]   = (diff_x_best-x)
        diff_image_x[i*2+1] = (diff_y_best-y)

    return diff_image_x

# find the norm of the given vector
def get_norm(coordinates):
    x = coordinates[0]
    y = coordinates[1]

    # normalize the vectors
    length_of_vector = math.sqrt(x**2+y**2)
    if length_of_vector == 0:
        print('length is zero, x: ', x, 'y: ', y)
    return np.array((-y/length_of_vector, x/length_of_vector))


# create vector t (translation) of same length as the shape
# with t_x, t_y repeated
def get_translation(t_x, t_y, length):
    t = np.array([t_x,t_y])
    t = np.tile(t, length)
    return t

# update a_x, a_y, t_x, t_y by multiplying the rotation/scale matrices and adding the translations
def update_parameters(alignment_parameters, diff_alignment_parameters):
    a_x   = alignment_parameters[0]
    a_y   = alignment_parameters[1]
    t_x   = alignment_parameters[2]
    t_y   = alignment_parameters[3]

    diff_a_x = diff_alignment_parameters[0]
    diff_a_y = diff_alignment_parameters[1]
    diff_t_x = diff_alignment_parameters[2]
    diff_t_y = diff_alignment_parameters[3]

    # build the rotation/scale matrices
    R      = build_matrix(a_x, a_y)
    diff_R = build_matrix(diff_a_x, diff_a_y)

    # the new rotation/scale matrix is found by multiplying the original with the updates
    new_R = np.dot(R, diff_R)

    # store in the list
    alignment_parameters[0] = new_R[0][0]
    alignment_parameters[1] = new_R[0][1]
    alignment_parameters[2] = t_x + diff_t_x
    alignment_parameters[3] = t_y + diff_t_y

    return alignment_parameters

# Build a standard rotation/scale matrix
def build_matrix(a_x, a_y):
    R = np.array(([a_x, -a_y], [a_y, a_x]))
    return R

# rotate and scale the shape according to the R matrix
def scale_and_rotate(shape, rotation_matrix):
    # mean center to rotate
    xes = shape[::2]
    yes = shape[1::2]
    mean_xes = np.mean(xes)
    mean_yes = np.mean(yes)
    mean_vector = get_translation(mean_xes, mean_yes, len(shape)/2)
    # the mean centered shape
    M = (shape - mean_vector)

    # rotate and scale one point at the time
    L = np.array(())
    for k in range(0,int(len(M)/2)):
        # the coordinates of the point to update
        l = np.array((M[k*2], M[k*2+1]))
        # rotating simply by dotting the rotation matrix and the cóordinate vector
        L = np.append(L, np.dot(rotation_matrix, l))

    # translate back to the original place in coordinate system
    L =  L  + mean_vector
    return L

def normalize_model(model_x, image, threshold, image_to_show):
    # split the x- and y coordinates of the model shape
    model_x_xes = model_x[::2]
    model_x_yes = model_x[1::2]

    # find the uppermost- and the lowest coordinate of the shape:
    # i.e points that satisfies the threshold
    upper_y = 0
    upper_x = 0
    lower_y = len(image)
    lower_x = 0
    for i in range(len(image)):
        for j in range(len(image[0])):
            if image[i][j] > threshold:
                if i < lower_y:
                    lower_y = i
                    lower_x = j
                if i > upper_y:
                    upper_y = i
                    upper_x = j

    # lowest coordinat will be the first point in the model
    # find the uppermost point
    mod_upper_y = 0
    mod_upper_x = 0
    mod_lower_y = model_x[1]
    mod_lower_x = model_x[0]
    for i in range(len(model_x_yes)):
        if model_x_yes[i] > mod_upper_y:
            mod_upper_y = model_x_yes[i]
            mod_upper_x = model_x_xes[i]

    # create a vector going through the uppermost - and lowest point
    v1 = np.array((upper_x - lower_x, upper_y - lower_y))
    v2 = np.array((mod_upper_x - mod_lower_x, mod_upper_y - mod_lower_y))

#   # plotting of the normals
#    plt.imshow(image_to_show,cmap = 'gray')
#    plt.plot([mod_lower_x, mod_upper_x], [mod_lower_y, mod_upper_y], marker='.')
#    plt.plot([lower_x, upper_x], [lower_y, upper_y], marker='.')
#    plt.plot(model_x_xes, model_x_yes)
#    plt.suptitle('Image x ', fontsize = 14)
#    plt.show()

    # find the angle between the two lines / the two shapes
    a_x_tmp, a_y_tmp = angle_between(v2, v1)

    # find the scaling that will make the model shape as 'long' as the leaf
    scale = np.linalg.norm(v1) / np.linalg.norm(v2)

    # update parameters
    a_x = scale * a_x_tmp
    a_y = scale * a_y_tmp

    return a_x, a_y, lower_x, lower_y


# Returns the unit vector of the given vector.
def unit_vector(vector):
    return vector / np.linalg.norm(vector)

#Returns the angle in radians between vectors 'v1' and 'v2'
def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.clip(np.dot(v1_u, v2_u), -1.0, 1.0), np.cross(v1_u, v2_u)

# normalize the image to have values between 0 and 255
def normalize_image(image):
    flat_image = image.flatten()
    # find min value and subtract this from all pixel values.
    min_value = min(flat_image)
    flat_image = flat_image - min_value
    # find max value and find out which scale is needed
    # such that pixel values run from 0 to 255
    max_value = max(flat_image)
    scale = 255 / max_value

    # update image
    for i in range(len(image)):
        for j in range(len(image[0])):
            image[i][j] = (image[i][j] - min_value) * scale

    return image


###############################


#with open('test_image_search_image.p', 'rb') as f:
#   image = pickle.load(f)
#
#with open('test_image_search_model.p', 'rb') as f:
#   asm_model = pickle.load(f)

#print('tests image: ', image)
#image_print = cv2.imread('../data/train/687.jpg', 0)
#image_print = cv2.imread('../data/test/15252.jpg', 0)
#image_print = cv2.imread('../data/train/891.jpg', 0)
#image_print = cv2.imread('../data/train/27.jpg', 0)
#image_print = cv2.imread('../data/train/60.jpg', 0)

#asm = construct()
# get all images
#test_list = os.listdir('../data/test/')

#max_count = len(test_list)/2

#i = 1

#test_images = test_list
#test_images = ['108138.xml']
#
#for test_image in test_images:
#    # make sure we only test each image one time
#    if test_image.endswith('.xml'):
#
#        print(str(i) + ' of ' + str(max_count))

        # remove the ending of the image
#        test_image = test_image[:-4]
#        if test_image == '15252' or test_image == '38507' or test_image ==  '108138' or test_image == '73780' or test_image == '24273'\
#        :#or test_image == '68284':
#            continue

#        print('image: ', test_image)
#        gray_image = parser.get_grayscale('../data/test/', test_image)

#        image_features, landmarks = image_search(asm_model, gray_image)


#image_print = parser.get_grayscale('../data/test/', '103527.jpg')
#
#plt.imshow(image_print,cmap = 'gray')
#plt.show()
#
#image_search(asm_model, image_print)


# PLOT HISTOGRAM
#    image_flat = image_diff.flatten()
#    for i in image_diff:
#        if i > 14:
#            print(i)
#    int_image_flat = np.array(image_flat, dtype=int)
#    print(max(int_image_flat))

#    diff_set = set(image_flat)
#    hist, bin_edges = np.histogram(image_diff, bins = range(len(diff_set)+1))
#    plt.bar(bin_edges[:-1], hist, width = 1)
#    plt.xlim(min(bin_edges), max(bin_edges))
#    print(bin_edges)
#    print(hist)
#    plt.show()

#    sys.exit(2)
