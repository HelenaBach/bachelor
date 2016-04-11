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
## arguments : binary image as numpy array
## returns   : numpy array of landmark coordinats (x, y tuples)
def landmark_setter(image):
	return

