# naive segmentation of the leaf from the background
# 
import numpy as np
import cv2

# segmentation of image based on otsu threshold selection method
## arguments : image in grayscale as numpy array
## returns   : binary image as numpy array
def otsu(image):

	threshold, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
	return binary_image

# place a set of 100 landmarks on image
## arguments : binary image as numpy array of leaf - background
## returns   : numpy array of landmark coordinats (x, y tuples)
def landmark_setter(image):
	# 
	# get the contour of the leaf
	img, contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	#            ^^ bruger ikke                    ^ for hirachy - CHAIN_APPROX_SIMPLE to let two points represent lines

	print(len(contours))
	print(contours[0])
	# expect only one contour - the leaf
	#contour = contours[0]
	h = cv2.drawContours(img, contours, -1, (0,255,0), 3)
	#cv2.imshow('img', img)
	#cv2.waitKey(0)
	#print(h)
	#print(img)
	#cv2.destroyAllWindows()
	# contour is a Numpy array of (x,y) coordinates of boundary points of the object.
	#print(contour)
	return contours

# inverse a binary image
def inverse_binary(image):
	for x in np.nditer(image, op_flags=['readwrite']):
		x[...] = 1-(x % 2)

	return image

import parser
image = parser.get_grayscale('../data/leafscan_selection/', '100261.jpg')
bin_im = otsu(image)
cv2.imshow('img', bin_im)
cv2.waitKey(0)
cv2.destroyAllWindows()
bin_im2 = inverse_binary(bin_im)
cv2.imshow('img', bin_im2)
cv2.waitKey(0)
cv2.destroyAllWindows()

#landmark_setter(bin_im2)