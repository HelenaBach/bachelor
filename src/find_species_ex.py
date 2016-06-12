import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np
import parser

try:
    class_id = sys.argv[1]
except:
    print('Pass on the class ID of the species')
    sys.exit(3)
try:
    n = sys.argv[2]
except:
    print('Give the number of images of the species you wish to see')
    sys.exit(3)


path = '../data/test/'
i = 0

with open('p_files/image_table.p', 'rb') as f:
    image_table = pickle.load(f)

#with open('p_files/test_table_ims_pc13_k3.p', 'rb') as f:
#    test_table = pickle.load(f)

with open('p_files/test_table_otsu_pc80_k3.p', 'rb') as f:
    test_table = pickle.load(f)

for img in test_table: #image_table:
    if img['class_id'] == class_id:
        image = parser.get_image(path, img['media_id'])
        implot = plt.imshow(image)
        plt.show()
        i += 1
        if i == int(n):
            sys.exit(3)
