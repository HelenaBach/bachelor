import numpy as np
import cv2
#import segmentation
import parser
import matplotlib.pyplot as plt
import pickle
import sys

with open('p_files/image_table_unaligned.p', 'rb') as f:
    image_table_unaligned = pickle.load(f)

path = '../../Data/train/'
file = '27.jpg'

image = parser.get_binary(path,file)

original = parser.get_image(path, file)

plt.imshow(image, cmap='gray')
plt.show()

#sys.exit(2)
landmarks = []

for img in image_table_unaligned:
	if img['media_id'] == '27':
		landmarks = np.copy(img['landmarks'])
		break

xes = landmarks[::2]
yes = landmarks[1::2]


implot = plt.imshow(original)
plt.plot(xes,yes, marker = 'o', c='r', lw=3.0)
plt.show()

implot = plt.imshow(image, cmap='gray')
plt.plot(xes,yes, c='r', lw=3.0)
plt.show()
