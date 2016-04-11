import cv2
import numpy as np
import math

# Compute the V_R_kl of all distances
# argument: image_table of all img_structs
# returns: n times n matirx of variances, where n is #landmarks
def compute_var_dist(image_table):

    matrix_stack = []

    for img in image_table:
        # (x,y)*n --> [x_0, y_0, x_1, ... , x_n-1, y_n-1]
        shape = img['landmarks']
        # number of (x,y)'s
        n = len(shape)/2
        # create empty n x n matrix to hold distances
        dist_matrix = np.zeros(n,n)
        for i in range(0,n):
            # the element from which the distance is measured
            k = np.array(shape[i*2], shape[i*2+1])
            for j in range(0,n):
                # the element to which the distance is measured
                l = np.array(shape[j*2], shape[j*2+1])
                # the distance from k to l
                dist_matrix[i][j] = np.linalg.norm(k-l)
                # stack each dist_matrix 'on top' of eachother
                matrix_stack = np.dstack(matrix_stack, dist_matrix)

    m = len(matrix_stack[0])
    var_matrix = np.zeros(m,m)

    # calculate the variance of each point distance
    for i in range(0, m):
        for j in range(0, m):
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
    # get all the x-coordinates of the shape
    for i in range(0, len(shape)/2):
        xses[i] = shape[i*2]
    summ = 0
    for j in range(0,len(xes)):
        weight = k_weight(var_matrix, j)
        summ += weight * xses[j]
    return summ

# Compute Y_i = sum^n-1_k=0 w_k * y_ik (31)
# arguments: the i'th shape, V_R_kl matrix
# returns: Y_i
def y_sum(shape, var_matrix):
    # get all the y-coordinates of the shape
    for i in range(0, len(shape)/2):
        yses[i] = shape[i*2+1]
    summ = 0
    for j in range(0,len(yses)):
        weight = k_weight(var_matrix, j)
        summ += weight * yses[j]
    return summ

# Compute Z = sum^n-1_k=0 w_k (x^2_nk + y^2_nk) (32)
# arguments: the i'th shape, V_R_kl matrix
# returns: Z
def z_sum(shape, var_matrix):
    summ = 0
    for i in range(0, len(shape)/2):
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
    for i in range(0, len(shape1)/2):
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
    for i in range(0, len(shape1)/2):
        weight = k_weight(var_matrix, i)
        x1x2 = shape1[i*2] * shape2[i*2]
        y1y2 = shape1[i*2+1] * shape2[i*2+1]
        summ += weight * (x1x2 - y1y2)
    return summ

# Aligning a pair of shapes by solving Ax=b
# arguments: shape1 the shape to align according to
#            shape2 the shape to be aligned
#            V_R_kl matirx
# returns: the new aligned points (landmarks) of shape2
def align_pair(shape1, shape2, var_matrix):
    # Fill the big matrix A of (30)
    A = np.zeros(4,4)
    # diagonal X_2
    for i in range(0,4):
        A[i][i] = x_sum(shape2, var_matrix)
    # -Y_2
    A[0][1] = A[3][2] = -(y_sum(shape2, var_matrix))
    # W
    A[0][2] = A[1][3] = w_sum(len(shape1)/2, var_matrix)
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
