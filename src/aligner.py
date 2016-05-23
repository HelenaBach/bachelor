import cv2
import numpy as np
import math
from table import image_table
import sys # debugging

# Compute the V_R_kl of all distances
# argument: image_table of all img_structs
# returns: n times n matirx of variances, where n is #landmarks
def compute_var_dist():
    # number of (x,y)'s
    n = int(len(image_table[0]['landmarks'])/2)
    # init dummy array to stack upon
    matrix_stack = np.ndarray((n,n))

    for img in image_table:
        # (x,y)*n --> [x_0, y_0, x_1, ... , x_n-1, y_n-1]
        shape = img['landmarks']
        # create empty n x n matrix to hold distances
        dist_matrix = np.ndarray((n,n))
        for i in range(0,n):
            # the element from which the distance is measured
            k = np.array((shape[i*2], shape[i*2+1]))
            for j in range(0,n):
                # the element to which the distance is measured
                l = np.array((shape[j*2], shape[j*2+1]))
                # the distance from k to l
                dist_matrix[i][j] = np.linalg.norm(k-l)
        # stack each dist_matrix 'on top' of eachother
        matrix_stack = np.dstack((matrix_stack, dist_matrix))
    # remove the dummy array
    matrix_stack = np.delete(matrix_stack, 0, axis=2)

    # init matrix of variances with uninitialized values
    var_matrix = np.ndarray((n,n))

    # calculate the variance of each point distance
    for i in range(0, n):
        for j in range(0, n):
            # find the variance of each point
            var_matrix[i][j] = np.var(matrix_stack[i][j])
    return var_matrix

# Compute weight for the k'th point (landmark)
# arguments: V_R_kl matrix, k'th point
# returns: weight
def k_weight(var_matrix, k):
    w = np.sum(var_matrix[k])
    w_k = math.pow(w, -1)
    return w_k

# Compute X_i = sum^n-1_k=0 w_k * x_ik (31)
# arguments: the i'th shape, V_R_kl matrix
# returns: X_i
def x_sum(shape, var_matrix):
    summ = 0
    for i in range(0,int(len(shape)/2)):
        weight = k_weight(var_matrix, i)
        summ += weight * shape[i*2]
    return summ

# Compute Y_i = sum^n-1_k=0 w_k * y_ik (31)
# arguments: the i'th shape, V_R_kl matrix
# returns: Y_i
def y_sum(shape, var_matrix):
    summ = 0
    for i in range(0,int(len(shape)/2)):
        weight = k_weight(var_matrix, i)
        summ += weight * shape[i*2+1]
    return summ

# Compute Z = sum^n-1_k=0 w_k (x^2_nk + y^2_nk) (32)
# arguments: the i'th shape, V_R_kl matrix
# returns: Z
def z_sum(shape, var_matrix):
    summ = 0
    for i in range(0, int(len(shape)/2)):
        # w_k
        weight = k_weight(var_matrix, i)
        # x^2_nk
        x_sq = math.pow(shape[i*2], 2)
        # y^2_nk
        y_sq = math.pow(shape[i*2+1], 2)
        summ += weight * (x_sq + y_sq)
    return summ

# Compute W = sum of all k_weights (32)
# arguments: n number of points (landmarks), V_R_kl
# returns: W
def w_sum(n, var_matrix):
    summ = 0
    for i in range(0, n):
        summ += k_weight(var_matrix, i)
    return summ

# Compute C_1 = sum^n-1_k=0 w_k (x_1k*x_2k + y_1k*y_2k)
# arguments: shape1, shape2, var_matrix
# returns: C_1
def c1(shape1, shape2, var_matrix):
    summ = 0
    for i in range(0, int(len(shape1)/2)):
        weight = k_weight(var_matrix, i)
        x1x2 = shape1[i*2] * shape2[i*2]
        y1y2 = shape1[i*2+1] * shape2[i*2+1]
        summ += weight * (x1x2 + y1y2)
    return summ

# Compute C_2 = sum^n-1_k=0 w_k (x_1k*x_2k - y_1k*y_2k)
# arguments: shape1, shape2, var_matrix
# returns: C_1
def c2(shape1, shape2, var_matrix):
    summ = 0
    for i in range(0, int(len(shape1)/2)):
        weight = k_weight(var_matrix, i)
        y1x2 = shape1[i*2+1] * shape2[i*2]
        x1y2 = shape1[i*2] * shape2[i*2+1]
        summ += weight * (y1x2 - x1y2)
    return summ

# Solve Ax=b (30)
# arguments: shape1 the shape to align according to
#            shape2 the shape to be aligned
#            V_R_kl matirx
# returns: the vector x = a_x, a_y, t_x, t_y
def solve_x(shape1, shape2, var_matrix):
    # Fill the big matrix A of (30)
    A = np.zeros((4,4))
    # diagonal X_2
    for i in range(0,4):
        A[i][i] = x_sum(shape2, var_matrix)
    # -Y_2
    A[0][1] = A[3][2] = -(y_sum(shape2, var_matrix))
    # W
    A[0][2] = A[1][3] = w_sum(int(len(shape1)/2), var_matrix)
    # Y_2
    A[1][0] = A[2][3] = abs(A[0][1])
    # Z
    A[2][0] = A[3][1] = z_sum(shape2, var_matrix)
    # Fill the vector b of (30)
    b = np.zeros(4)
    # X_1
    b[0] = x_sum(shape1, var_matrix)
    # Y_2
    b[1] = y_sum(shape1, var_matrix)
    # C_1
    b[2] = c1(shape1, shape2, var_matrix)
    # C_2
    b[3] = c2(shape1, shape2, var_matrix)
    # solve for x = a_x, a_y, t_x, t_y)
    x = np.linalg.solve(A,b)
    return x

# Align shape2 (x_2) by mapping x_2 onto M(x_2)+t
# arguments: the shape to be aligned, x vector containing s, theta and (t_x, t_y)
# returns: The aligned shape
def align_pair(shape2, x):
    a_x = x[0] # s cos theta
    a_y = x[1] # s sin theta
    t_x = x[2]
    t_y = x[3]

    n = int(len(shape2)/2)
    t = np.array([t_x,t_y])
    # create vector t (translation) of same lenght as shape2
    # with t_x, t_y repeated
    t = np.tile(t, n)
    # initiate a vector of same dimentions and values as shape2
    M = np.copy(shape2)
    # rotate and scale shape2 (now know as M)
    for k in range(0,n):
        M[k*2]   = (a_x * M[k*2]) - (a_y * M[k*2+1])
        M[k*2+1] = (a_y * M[k*2]) + (a_x * M[k*2+1])
    # translate by t
    M = M + t
    return M

# Align all shapes with the given shape
# arguments: the image table with all image dicts, the shape with which all
#            others are to be aligned
# returns: var_matrix (kind of a hack)
def align_all_shapes(mean_shape):
    var_matrix = compute_var_dist()
    for img in image_table:
        shape = img['landmarks']
        x = solve_x(mean_shape, shape, var_matrix)
        img['landmarks'] = align_pair(shape, x)
    return var_matrix

# Calculate the mean shape from the aligned shapes
# returns: the mean shape
def mean_shape():
    shape_stack = image_table[0]['landmarks']
    for img in image_table[1:]:
        shape = img['landmarks']
        shape_stack = np.dstack((shape_stack, shape))

    shape_length = len(shape_stack[0])
    mean_shape = np.zeros(shape_length)
    for i in range(0, shape_length):
        mean_shape[i] = np.mean(shape_stack[0][i])

    return mean_shape

# Normalize the meanshape arcording to the first shape in the set
#  - this is the 'easy' solution. One could some other normalizing default
# arguments: the first shape in the set, the current meanshape, and the V_R_kl matrix
# returns: the normalized meanshape.
def normalize_mean(shape1, mean_shape, var_matrix):
    x = solve_x(shape1, mean_shape, var_matrix)
    norm_mean = align_pair(mean_shape, x)
    return norm_mean

# The align algorithm of Cootes
def the_real_aligner():
    # the first shape is saved to be used for normalizing mean
    shape1 = np.copy(image_table[0]['landmarks'])
    # rotate, scale and translate each shape to align with the first shape
    var_matrix = align_all_shapes(shape1)
    # initial previous mean_shape - dummy 1x200 vector of zeros
    prev_mean = np.array((0))
    prev_mean = np.tile(prev_mean, 200)
    for i in range(10):
        print('aligner iteration: ' + str(i))
        mean = mean_shape()
        # check if prev_mean and mean is 'equal' - does the process converge
        diff = sum(abs(prev_mean-mean))
        print('sum of diff: ' + str(diff))
        if diff < 50 or i == 9:
            print('sum of diff: ' + str(diff))
            return mean, var_matrix
        new_mean = normalize_mean(shape1, mean, var_matrix)
        align_all_shapes(new_mean)
        prev_mean = mean
