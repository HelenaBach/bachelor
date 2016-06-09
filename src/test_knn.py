### Statistics and evaluation of knn prediction

import numpy as np
import pickle
import sys
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
#with open ('test_table.p', 'rb') as f:
#	test_table = pickle.load(f)


try:
    seg = sys.argv[1]
except:
    print("Say 'otsu' or 'ims' to specify which segmentation to use. ")
    sys.exit(2)

try:
	p = sys.argv[2]
except:
	print('specify number of PCs')
	sys.exit(2)

try:
	k = sys.argv[3]
except:
	print('specify k')
	sys.exit(2)

## show accuracy of classifier on the testset using cross validation with the given amount of folds
### arguments :
### return    : the accuracy of the classifier
def knn_show_accuracy():
    with open ('p_files/ROC_table_' + str(seg) + '_pc' + str(p) + '_k' + str(k) + '.p', 'rb') as f:
        ROC_table = pickle.load(f)

    number_of_instances = sum(item['number'] for item in ROC_table.values())
    # HVAD BRUGER VI DEM HER TIL????
    #sum_fp = sum(item['fp'] for item in ROC_table.values())
    #sum_tp = sum(item['tp'] for item in ROC_table.values())

    # for each spieces
    for specie in ROC_table:
        # False Positive
        FP = ROC_table[specie]['fp']
        # True Positive
        TP = ROC_table[specie]['tp']
        # Number of instances of this specie
        P  = ROC_table[specie]['number']
        # All instances that is not specie
        N  = number_of_instances - P
        # False Negative
        FN = P - TP
        # True Negative
        TN = (P + N) - (TP + FP + FN)
        #find fp rate fp/N
        fp_rate = FP / N
        # find tp rate tp/P - RECALL
        tp_rate = TP / P
        # find precision and find F-measure (WE DO NOT WANT TO DIVIDE BY ZERO!)
        precision = 0
        f_measure = 0
        if not TP == 0:
            precision = TP/(TP + FP)
            f_measure = 2/((1/precision)+(1/tp_rate))

        #print(specie, ' accuracy: ', accuracy)
        ROC_table[specie]['fp_rate']   = "{0:.3f}".format(fp_rate)
        ROC_table[specie]['tp_rate']   = "{0:.3f}".format(tp_rate) # RECALLLLL
        ROC_table[specie]['precision'] = "{0:.3f}".format(precision)
        ROC_table[specie]['f_measure'] = "{0:.3f}".format(f_measure)


    with open('p_files/ROC_table_' + str(seg) + '_pc' + str(p) + '_k' + str(k) + '.p', 'wb') as f:
        pickle.dump(ROC_table, f)


def get_accuracies():
    with open ('p_files/ROC_table_' + str(seg) + '_pc' + str(p) + '_k' + str(k) + '.p', 'rb') as f:
        ROC_table = pickle.load(f)

    with open ('p_files/species_stats.p', 'rb') as f:
        species_table = pickle.load(f)

    sort_by_specie = sorted(ROC_table.items(), key=lambda item:int(item[0]))
    print(sort_by_specie[0])
    print('Specie - Number of Train - Number of Test - Recall - Precision - F-measure')
    for specie_tup in sort_by_specie:
        specie = specie_tup[0]
        print(specie, ' & ', species_table[specie], ' & ', ROC_table[specie]['number'], ' & ', ROC_table[specie]['tp_rate'], ' & ', ROC_table[specie]['precision'], ' & ', ROC_table[specie]['f_measure'], '\\\\')


def plot_ROC():
    with open ('ROC_table.p', 'rb') as f:
        ROC_table = pickle.load(f)

    xes = [item['fp_rate'] for item in ROC_table.values()]
    yes = [item['tp_rate'] for item in ROC_table.values()]
    labels = ROC_table.keys()

    plt.subplots_adjust(bottom=0.1)
    plt.plot([0,1],[0,1], ls="--")
    plt.scatter(xes, yes, marker = 'o')
    for label, x, y in zip(labels, xes, yes):
        plt.annotate(
            label,
            xy = (x, y), xytext = (0.1, 0.1),
            textcoords = 'offset points', ha = 'right', va = 'bottom',
            #bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
    plt.suptitle('ROC graph ish', fontsize = 14)
#    ax.set_xlabel('Principal Components')
#    ax.set_ylabel('Percentage of Variance')
    plt.show()

knn_show_accuracy()
get_accuracies()
#plot_ROC()
