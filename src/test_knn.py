### Statistics and evaluation of knn prediction
import parser
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
    # print(sort_by_specie[0])

    sort_by_number_of_train = sorted(species_table.items(), key=lambda item:int(item[1]), reverse=True)

    sort_by_fmeasure = sorted(ROC_table.items(), key=lambda k_v: k_v[1]['f_measure'], reverse=True)

    print('Specie & Number of Train & Number of Test & Recall & Precision & F-measure \\\\')
    for specie_tup in sort_by_fmeasure:
        specie = specie_tup[0]
       # number = specie_tup[1]
        print(specie, ' & ', species_table[specie], ' & ', ROC_table[specie]['number'], ' & ', ROC_table[specie]['tp_rate'], ' & ', ROC_table[specie]['precision'], ' & ', ROC_table[specie]['f_measure'], '\\\\')

def get_sorted_species():
    with open ('p_files/ROC_table_' + str(seg) + '_pc' + str(p) + '_k' + str(k) + '.p', 'rb') as f:
        ROC_table = pickle.load(f)
    with open ('p_files/species_stats.p', 'rb') as f:
        species_table = pickle.load(f)
    with open ('p_files/image_table.p', 'rb') as f:
        image_table = pickle.load(f)

    path = '../../data/train/'

    sort_by_number_of_train = sorted(species_table.items(), key=lambda item:int(item[1]), reverse=True)
    class_id_list = []
    specie_list = []
    recall_list = []
    precision_list = []
    f_list = []
    for specie_tup in sort_by_number_of_train:
        specie = specie_tup[0]
        for img in image_table:
            if img['class_id'] == specie:
                filename = img['media_id'] + '.xml'
                specie_name = parser.get_specie_name(path, filename)
                specie_list.append(specie_name)
                break
        recall_list.append(ROC_table[specie]['tp_rate'])
        precision_list.append(ROC_table[specie]['precision'])
        f_list.append(ROC_table[specie]['f_measure'])
        class_id_list.append(specie)

    print('class_ider', class_id_list)
    print('specie names', specie_list)
    print('recalls', recall_list)
    print('precisions', precision_list)
    print('F measure', f_list)


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

def averages(seg, p, k):
    with open ('p_files/ROC_table_' + str(seg) + '_pc' + str(p) + '_k' + str(k) + '.p', 'rb') as f:
        ROC_table = pickle.load(f)

    recall_sum = sum(float(item['tp_rate']) for item in ROC_table.values())
    recall_sum_percentage = recall_sum * 100.0
    recall_average = recall_sum_percentage / len(ROC_table)

    #print('Average of recall for ', seg, 'with ', p, ' PCs and ', k, ' neighbours ', recall_average)

    precision_sum = sum(float(item['precision']) for item in ROC_table.values())
    precision_sum_percentage = precision_sum * 100.0
    precision_average = precision_sum_percentage / len(ROC_table)

    #print('Average of precision for ', seg, 'with ', p, ' PCs and ', k, ' neighbours ', precision_average)

    f_sum = sum(float(item['f_measure']) for item in ROC_table.values())
    f_sum_percentage = f_sum * 100.0
    f_average = f_sum_percentage / len(ROC_table)

    #print('Average of F-measure for ', seg, 'with ', p, ' PCs and ', k, ' neighbours ', f_average)
    return recall_average, precision_average, f_average





#knn_show_accuracy()
#get_accuracies()
#plot_ROC()

#averages()
get_sorted_species()
