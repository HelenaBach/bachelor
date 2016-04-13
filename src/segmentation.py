# naive segmentation of the leaf from the background
# 
import numpy as np
import cv2
import math

# segmentation of image based on otsu threshold selection method
## arguments : image in grayscale as numpy array
## returns   : binary image as numpy array
def otsu(image):

	threshold, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
	return binary_image

# place a set of 100 landmarks on image
## arguments : binary image as numpy array of leaf - background
## returns   : numpy array of landmark coordinats (x, y tuples)
def landmark_setter(image, gray_image):
	
	# need to invert the image to use findContours
	image = np.invert(image)

	# get the contour of the leaf
	img, contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	#^^             ^^ bruger ikke                      ^ for hirachy - CHAIN_APPROX_SIMPLE to let two points represent lines

	# each contour is a Numpy array of (x,y) coordinates of boundary points of the leaf.
	# expect only one contour - the leaf - should be the biggest if more contours is found
	contour = contours[0]
	
	# want to uniformly place 100 point along the contour
	pix_pr_landmark = math.floor(len(contour) / 100)
	# floor is used to ensure that we can put 100 points

	# contour have the format [[[x0, y0]], [[x1, y1]], ... ,[[xN, yN]]] -> [[x0, y0], [x1, y1], ... ,[xN, yN]]
	contour_single = [i for sub in contour for i in sub]

	# get a point every 'pix_pr_landmark' and reshape to 1D array  
	# [[x0, y0], [x1, y1], ... ,[xN, yN]] -> [x0, y0, x1, y1, ... ,xN, yN]
	landmarks = [i for sub in contour_single[::pix_pr_landmark] for i in sub]

	# make sure that only 100 landmarks are chosen. x and y for each point -> 200 elements
	landmarks = landmarks[:200]
	return landmarks