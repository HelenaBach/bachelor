import os
import cv2
import segmentation
import parser
from table import image_table

path = '../data/leafscan/'
images = os.listdir(path)[:100]

for image in images:
	if image.endswith('.xml'):
		im_struct = parser.init_image(path, image)

		image_table.append(im_struct)

for im_struct in image_table:
    # get image as grayscale
    img = parser.get_grayscale(path, im_struct['media_id'])
    # gør Otsu agtige ting
    binary = segmentation.otsu(img)
	
	# get landmarks
	landmarks = segmentation.landmark_setter(binary, img)
	# update the image table
	im_struct['landmarks'] = landmarks




