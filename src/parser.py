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
	file_name = file + '.xml'
	if file[-4:-3] == '.':
		file_name = file[:-4] + '.xml'


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
	file_name = file + '.jpg'
	if file[-4:-3] == '.':
		file_name = file[:-4] + '.jpg'

	# read image as is (-1)
	image = cv2.imread(path + file_name, -1)
	if image == None:
		print('no image found')
		sys.exit(3)
	return image

# get gray scaled image as a numpy array
# arguments: the relative path to file directory, file
# returns  : gray scaled image as a numpy array
def get_grayscale(path, file):
	# correct file extension
	file_name = file + '.jpg'
	if file[-4:-3] == '.':
		file_name = file[:-4] + '.jpg'

	# read image in grayscale (0)
	image = cv2.imread(path + file_name, 0)
	if image == None:
		print('no image found')
		sys.exit(3)
	return image

# get segmentated image as a numpy array
# arguments: the relative path to file directory, file
# returns  : binary image as a numpy array
def get_binary(path, file):
	# should we call otsu every time or store the segmented image?
	if image.endswith('.jpg'):
	    # indlæs billedet i greyscale (det er det 0 betyder)
	    img = cv2.imread(path + file,0)
	    if img == None:
	    	print('no image found')
	    	sys.exit(3)
	    # gør Otsu agtige ting
	    ret,thr =cv2.threshold(img,0,255,cv2.THRESH_OTSU)
	return
