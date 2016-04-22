import os
import sys
import cv2
import segmentation
import matplotlib.pyplot as plt
import parser
from table import image_table
import asm


try:
    # get the name of the csv file of the reviews
    path = sys.argv[1]
except:
    print('The path of the image directory should be passed as argument to this script')
    sys.exit(2)

#path = '../../Data/leafscan/'
images = os.listdir(path)[:50]

#images = ['28366.jpg', '64400.xml', '75225.jpg', '76208.jpg', '17258.xml', '99000.xml', '83128.jpg', '50538.jpg', '15252.xml']

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
        xes = landmarks[::2]
        yes = landmarks[1::2]
        plt.plot(xes,yes)

# plot the shapes

plt.show()
# align the dataset
asm.the_real_aligner()

# put aligned landmarks in new plot_list, so that we can plot them

for im_struct in image_table:
        landmarks = im_struct['landmarks']
        xes = landmarks[::2]
        yes = landmarks[1::2]
        plt.plot(xes,yes)

# plot aligned shapes

plt.show()
