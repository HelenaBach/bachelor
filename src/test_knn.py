### Statistics and evaluation of knn prediction

import numpy as np 
import pickle
import sys 
from matplotlib import pyplot as plt 
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
#with open ('test_table.p', 'rb') as f:
#	test_table = pickle.load(f)

with open ('ROC_table.p', 'rb') as f:
    ROC_table = pickle.load(f)

## show accuracy of classifier on the testset using cross validation with the given amount of folds
### arguments : 
### return    : the accuracy of the classifier 
def knn_show_accuracy():
    with open ('ROC_table.p', 'rb') as f:
        ROC_table = pickle.load(f)

    number_of_instances = sum(item['number'] for item in ROC_table.values())
    sum_fp = sum(item['fp'] for item in ROC_table.values())
    sum_tp = sum(item['tp'] for item in ROC_table.values())

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
        # find tp rate tp/P
        tp_rate = TP / P
        # find accuracy
        accuracy = (TP + TN) / (P + N)
        #print(specie, ' accuracy: ', accuracy)
        ROC_table[specie]['fp_rate']  = fp_rate
        ROC_table[specie]['tp_rate']  = tp_rate
        ROC_table[specie]['accuracy'] = accuracy
        if TP == 0:
            precision = 0
        else:
            precision = TP/(TP + FP)
        if P > 4:
            print('number images: ', P)
            print('recall: ', tp_rate)
            print('precision ', precision)

    with open('ROC_table.p', 'wb') as f:
        pickle.dump(ROC_table, f)


def get_accuracies():
    with open ('ROC_table.p', 'rb') as f:
        ROC_table = pickle.load(f)

    with open ('species_stats.p', 'rb') as f:
        species_table = pickle.load(f)
    print('Specie - Number of Train - Number of Test - Accuracy')
    for specie in ROC_table:
        print(specie, ' & ', species_table[specie], ' & ', ROC_table[specie]['number'], ' & ', ROC_table[specie]['accuracy'],  '\\\\')


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
#get_accuracies()
plot_ROC()