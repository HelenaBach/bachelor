import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np
import parser

def find_species_ex(class_id, n, prnt='p'):
    path = '../../data/train/'
    i = 0

    with open('p_files/image_table.p', 'rb') as f:
        image_table = pickle.load(f)

    #with open('p_files/test_table_ims_pc13_k3.p', 'rb') as f:
    #    test_table = pickle.load(f)

    with open('p_files/test_table_ims_pc28_k9.p', 'rb') as f:
        test_table = pickle.load(f)

    for img in image_table:
        if img['class_id'] == class_id:
            image = parser.get_image(path, img['media_id'])
            implot = plt.imshow(image)
            if prnt == 'p':
                plt.show()
            else:
                #fig1 = plt.gcf()
                plt.savefig('plots/' + prnt + class_id + '_' + str(i) + '.png', bbox_inches='tight')   # save the figure to file
                plt.close()
            i += 1
            if i == int(n):
                return

#def find_species_ex_media_id(meadia_id, n):
#    path = '../data/train/'
#    i = 0
#
#    with open('p_files/image_table.p', 'rb') as f:
#        image_table = pickle.load(f)
#
#    #with open('p_files/test_table_ims_pc13_k3.p', 'rb') as f:
#    #    test_table = pickle.load(f)
#
#    with open('p_files/test_table_otsu_pc80_k3.p', 'rb') as f:
#        test_table = pickle.load(f)
#
#    for img in image_table:
#        print(img['class_id'])
#        print('class: ', class_id)
#        if img['class_id'] == class_id:
#            image = parser.get_image(path, img['media_id'])
#            implot = plt.imshow(image)
#            plt.show()
#            i += 1
#            if i == int(n):
#                return
