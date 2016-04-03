# Parse data from leafscan 
import xml.etree.ElementTree as etree
import os
from shutil import copy
import numpy as np
import cv2

# instantiate an image object.
## arguments: the relative path to file directory, file
## returns: the image dict
def init_image(path, file):
	# correct file extension
	file_name = file[:-3] + '.xml'

	img_struct = {}
	tree = etree.parse(path + file_name)
	root = tree.getroot()

	# create image fields 
	media_id       = root.find('MediaId').text
	class_id       = root.find('ClassId').text
	landmarks      = np.array([]) # <- skal der mon være en tom liste her?
	feature_vector = np.array([])
	# instantiate image fields 
	img_struct['media_id']       = media_id
	img_struct['class_id']       = class_id
	img_struct['landmarks']      = landmarks
	img_struct['feature_vector'] = feature_vector

	return img_struct

# get image as a numpy array
# arguments: the relative path to file directory, file
# returns  : image as a numpy array
def get_image(path, file):
	# correct file extension
	file_name = file[:-3] + '.jpg'
	# read image as is (-1)
	image = cv2.imread(path + file_name, -1)
	return image

# get gray scaled image as a numpy array
# arguments: the relative path to file directory, file
# returns  : gray scaled image as a numpy array
def get_grayscale(path, file):
	# correct file extension
	file_name = file[:-3] + '.jpg'
	# read image in grayscale (0)
	image = cv2.imread(path + file_name, 0)
	return image

# get segmentated image as a numpy array
# arguments: the relative path to file directory, file
# returns  : binary image as a numpy array
def get_segmented_image():
	# should we call otsu every time or store the segmented image?
	return  