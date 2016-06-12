### Statistics and evaluation of knn prediction
import parser
import numpy as np
import pickle
import sys
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import find_species_ex_func
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

    path = '../data/train/'

    sort_by_number_of_train = sorted(species_table.items(), key=lambda item:int(item[1]), reverse=True)
    class_id_list = []
    specie_list = []
    recall_list = []
    precision_list = []
    number = []
    f_list = []
    for specie_tup in sort_by_number_of_train:
        specie = specie_tup[0]
        number.append(str(specie_tup[1]))
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

    #print('class_ider', class_id_list)
    #print('specie names', specie_list)
    #print('recalls', recall_list)
    #print('precisions', precision_list)
    #print('F measure', f_list)
    print ('Number', number)

def plot_ROC():
    with open ('p_files/ROC_table_' + str(seg) + '_pc' + str(p) + '_k' + str(k) + '.p', 'rb') as f:
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

def plot_PR():
    with open ('p_files/ROC_table_' + str(seg) + '_pc' + str(p) + '_k' + str(k) + '.p', 'rb') as f:
        ROC_table = pickle.load(f)

    xes = [item['tp_rate'] for item in ROC_table.values()]
    yes = [item['precision'] for item in ROC_table.values()]
    label = ROC_table.keys()
    #print(len(xes))
    ax = plt.gca()
    plt.subplots_adjust(bottom=0.1)
    #plt.plot([0,1],[0,0.1382516352796756619], ls="--")
    ax.scatter(xes, yes, marker = '.')
    for i, txt in enumerate(label):
        if float(yes[i]) < 0.25 and float(xes[i]) > 0.4: #txt in interesting and
        #if xes[i] > 100 and float(yes[i]) < 0.3 or xes[i] > 200:
            print(txt)
            #ax.annotate(parser.get_specie_name('../data/train/', str(species_name[txt][0]) + '.xml'), (xes[i],yes[i]))
            ax.annotate(txt, (xes[i],yes[i]))
    #plt.suptitle('PR graph ', fontsize = 14)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    #plt.savefig('plots/PR_' + str(seg) + '.png')   # save the figure to file
    #plt.close()
    plt.show()

def plot_Precision():
    with open ('p_files/ROC_table_' + str(seg) + '_pc' + str(p) + '_k' + str(k) + '.p', 'rb') as f:
        ROC_table = pickle.load(f)

    xes = [item['number'] for item in ROC_table.values()]
    yes = [item['precision'] for item in ROC_table.values()]
    labels = ROC_table.keys()
    #print(len(xes))
    plt.subplots_adjust(bottom=0.1)
    plt.plot([0,max(xes)],[0.2655605381165919, 0.2655605381165919], ls="--")
    plt.scatter(xes, yes, marker = '.')
    #for label, x, y in zip(labels, xes, yes):
        #plt.annotate(
        #    label,
        #    xy = (x, y), xytext = (0.1, 0.1),
        #    textcoords = 'offset points', ha = 'right', va = 'bottom',
        #    #bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        #    arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
    axes = plt.gca()
    #plt.suptitle('ROC graph ish', fontsize = 14)
    axes.set_xlabel('number of species')
    axes.set_ylabel('Precision')
    plt.show()

def plot_Precision_inv():
    with open ('p_files/ROC_table_' + str(seg) + '_pc' + str(p) + '_k' + str(k) + '.p', 'rb') as f:
        ROC_table = pickle.load(f)

    xes = [item['number'] for item in ROC_table.values()]
    yes = [1.0-float(item['precision']) for item in ROC_table.values()]
    label = ROC_table.keys()
    #print(len(xes))
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.1)
    #plt.plot([0,max(xes)],[0.2655605381165919, 0.2655605381165919], ls="--")
    plt.scatter(xes, yes, marker = '.')
    #ax = plt.gca()
    for i, txt in enumerate(label):
        if txt in interesting:
        #if xes[i] > 100 and float(yes[i]) < 0.3 or xes[i] > 200:
            print(txt)
            #ax.annotate(parser.get_specie_name('../data/train/', str(species_name[txt][0]) + '.xml'), (xes[i],yes[i]))
            ax.annotate(txt, (xes[i],yes[i]))
    #plt.suptitle('ROC graph ish', fontsize = 14)
    ax.set_xlabel('number of species')
    ax.set_ylabel('Precision')
    plt.show()

def plot_Recall():
    print('REEECALLL')
    with open ('p_files/ROC_table_' + str(seg) + '_pc' + str(p) + '_k' + str(k) + '.p', 'rb') as f:
        ROC_table = pickle.load(f)

    with open ('p_files/species_stats.p', 'rb') as f:
        species_table = pickle.load(f)
    #xes = [item['number'] for item in ROC_table.values()]
    #yes = [item['tp_rate'] for item in ROC_table.values()]
    #labels = ROC_table.keys()

    xes = []#[item['number'] for item in ROC_table.values()]
    yes = []#[item['fp'] for item in ROC_table.values()]
    label = []
    high_FP = []
    for specie in ROC_table:
        #if species_table[specie] < 150:
        xes.append(species_table[specie])
        yes.append(ROC_table[specie]['tp_rate'])
        label.append(specie)
        if float(ROC_table[specie]['tp_rate']) < 0.4 and species_table[specie] > 190:
            print(specie)
        #if ROC_table[specie]['tp_rate'] == 0:
        #    high_FP.append((specie, ROC_table[specie]['fp']))
    sys.exit(4)
    #print(len(xes))
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.1)
    #plt.plot([0,max(xes)],[0.24845291479820627,0.24845291479820627], ls="--")
    ax.scatter(xes, yes, marker = '.')
    print('Recall')
    for i, txt in enumerate(label):
        if txt in interesting:
        #if xes[i] > 100 and float(yes[i]) < 0.3 or xes[i] > 200:
            print(txt)
            #ax.annotate(parser.get_specie_name('../data/train/', str(species_name[txt][0]) + '.xml'), (xes[i],yes[i]))
            ax.annotate(txt, (xes[i],yes[i]))
    ax = plt.gca()
    #plt.suptitle('Recall', fontsize = 14)
    ax.set_xlabel('Number of Instances')
    ax.set_ylabel('Recall')
    #plt.savefig('plots/recall_' + str(seg) + '.png')   # save the figure to file
    #plt.close()
    plt.show()

def plot_fmeasure():
    with open ('p_files/ROC_table_' + str(seg) + '_pc' + str(p) + '_k' + str(k) + '.p', 'rb') as f:
        ROC_table = pickle.load(f)

    with open ('p_files/species_stats.p', 'rb') as f:
        species_table = pickle.load(f)
    #xes = [item['number'] for item in ROC_table.values()]
    #yes = [item['tp_rate'] for item in ROC_table.values()]
    #labels = ROC_table.keys()

    xes = []#[item['number'] for item in ROC_table.values()]
    yes = []#[item['fp'] for item in ROC_table.values()]
    label = []
    high_FP = []
    really_bad = []
    for specie in ROC_table:
        xes.append(species_table[specie])
        yes.append(ROC_table[specie]['f_measure'])
        label.append(specie)
        if float(ROC_table[specie]['f_measure']) == 0.0:
            really_bad.append((specie, species_table[specie], ROC_table[specie]['f_measure']))

    #xes = [item['number'] for item in ROC_table.values()]
    #yes = [item['f_measure'] for item in ROC_table.values()]
    #labels = ROC_table.keys()
    really_bad.sort(key=lambda tup: tup[1], reverse=True)
    print(really_bad)
    print(len(really_bad))

    #print(len(xes))
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.1)
    #plt.plot([0,max(xes)],[0.24845291479820627,0.24845291479820627], ls="--")
    ax.scatter(xes, yes, marker = '.')
    print('F measure')
    for i, txt in enumerate(label):
        if txt in interesting and xes[i] > 200 or float(yes[i]) > 0.42:
        #if xes[i] > 100 and float(yes[i]) < 0.3 or xes[i] > 200:
            print(txt)
            #ax.annotate(parser.get_specie_name('../data/train/', str(species_name[txt][0]) + '.xml'), (xes[i],yes[i]))
            ax.annotate(txt, (xes[i],yes[i]))
    #axes = plt.gca()
    #plt.suptitle('F measure', fontsize = 14)
    ax.set_xlabel('Number of Instances')
    ax.set_ylabel('F-measure')
    minor_ticks = np.arange(0, 350, 10)
    ax.set_xticks(minor_ticks, minor=True)
    #plt.savefig('plots/fmeasure_' + str(seg) + '.png')   # save the figure to file
    #plt.close()
    plt.show()


def plot_FP():
    with open ('p_files/ROC_table_' + str(seg) + '_pc' + str(p) + '_k' + str(k) + '.p', 'rb') as f:
        ROC_table = pickle.load(f)
    with open ('p_files/species_stats.p', 'rb') as f:
        species_table = pickle.load(f)

    with open ('p_files/species.p', 'rb') as f:
        species_name = pickle.load(f)

    xes = []#[item['number'] for item in ROC_table.values()]
    yes = []#[item['fp'] for item in ROC_table.values()]
    label = []
    high_FP = []
    for specie in ROC_table:
        xes.append(species_table[specie])
        yes.append(ROC_table[specie]['fp'])
        label.append(specie)
        if ROC_table[specie]['fp'] > 50:
            high_FP.append((specie, ROC_table[specie]['fp']))

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.1)
    #ax.plot([0,max(xes)],[40,40], ls="--")
    ax.scatter(xes, yes, marker = '.')
    print('FP')
    for i, txt in enumerate(label):
        if yes[i] > 50 and not txt == '5128':# txt in  interesting or
        #if yes[i] > 40 or xes[i] > 200:
            print(txt)
            #ax.annotate(parser.get_specie_name('../data/train/', str(species_name[txt][0]) + '.xml'), (xes[i],yes[i]))
            ax.annotate(txt, (xes[i],yes[i]))
    #for label, x, y in zip(labels, xes, yes):
        #plt.annotate(
        #    label,
        #    xy = (x, y), xytext = (0.1, 0.1),
        #    textcoords = 'offset points', ha = 'right', va = 'bottom',
        #    #bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        #    arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
#    plt.suptitle('False Positive', fontsize = 14)
    ax.set_xlabel('Number of Instances')
    ax.set_ylabel('Number of Instances Incorrectly Labeled')
    #plt.savefig('plots/fp_' + str(seg) + '.png')   # save the figure to file
    #plt.close()
    plt.show()


 #FN/NUMBER = 1-RECALL -> USE THIS INSTEAD!
def plot_FN():
    with open ('p_files/ROC_table_' + str(seg) + '_pc' + str(p) + '_k' + str(k) + '.p', 'rb') as f:
        ROC_table = pickle.load(f)
    with open ('p_files/species_stats.p', 'rb') as f:
        species_table = pickle.load(f)

    with open ('p_files/species.p', 'rb') as f:
        species_name = pickle.load(f)

    yes = []
    xes = []
    label = []
    high_FP = []
    for specie in ROC_table:
        xes.append(species_table[specie])
        yes.append(ROC_table[specie]['number'] - ROC_table[specie]['tp'])
        label.append(specie)
    #    if ROC_table[specie]['fp'] > 40:
    #        high_FP.append((specie, ROC_table[specie]['fp']))

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.1)
    ax.plot([0,max(xes)],[40,40], ls="--")
    ax.scatter(xes, yes, marker = '.')
    for i, txt in enumerate(label):
        if yes[i] > 40:
            #print(txt)
            #ax.annotate(parser.get_specie_name('../data/train/', str(species_name[txt][0]) + '.xml'), (xes[i],yes[i]))
            ax.annotate(txt, (xes[i],yes[i]))
    #for label, x, y in zip(labels, xes, yes):
        #plt.annotate(
        #    label,
        #    xy = (x, y), xytext = (0.1, 0.1),
        #    textcoords = 'offset points', ha = 'right', va = 'bottom',
        #    #bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        #    arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
    plt.suptitle('False Negative', fontsize = 14)
#    ax.set_xlabel('Principal Components')
#    ax.set_ylabel('Percentage of Variance')
    plt.show()
    yes.sort(reverse=True)
    print(yes[:6])


def gets_classified_as(class_id):
    with open('p_files/test_table_' + str(seg) + '_pc' + str(p) + '_k' + str(k) + '.p', 'rb') as f:
        test_table = pickle.load(f)

    classify = []
    classifier = []
    for im_struct in test_table:
        if im_struct['label_candidates'][0][0] == class_id:
            classify.append((im_struct['class_id'], im_struct['media_id'], im_struct['landmarks']))
 #       if class_id in im_struct['label_candidates']:
 #           classifier.append((im_struct['class_id'], im_struct['media_id'], im_struct['landmarks']))

    return classify

def gets_classified_by(class_id):
    with open('p_files/test_table_' + str(seg) + '_pc' + str(p) + '_k' + str(k) + '.p', 'rb') as f:
        test_table = pickle.load(f)


def plot_TP():
    with open ('p_files/ROC_table_' + str(seg) + '_pc' + str(p) + '_k' + str(k) + '.p', 'rb') as f:
        ROC_table = pickle.load(f)
    with open ('p_files/species_stats.p', 'rb') as f:
        species_table = pickle.load(f)

    with open ('p_files/species.p', 'rb') as f:
        species_name = pickle.load(f)

    xes = []#[item['number'] for item in ROC_table.values()]
    yes = []#[item['fp'] for item in ROC_table.values()]
    label = []
    low_TP = []
    for specie in ROC_table:
        xes.append(species_table[specie])
        yes.append(ROC_table[specie]['tp_rate'])
        label.append(specie)
        if float(ROC_table[specie]['tp_rate']) < 0.3 and species_table[specie] > 100:
            #print(ROC_table[specie]['tp_rate'])
            low_TP.append((specie, ROC_table[specie]['tp']))

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.1)
    ax.plot([0,max(xes)],[0.3,0.3], ls="--")
    ax.scatter(xes, yes, marker = '.')
    for i, txt in enumerate(label):
        if txt in interesting:
        #if float(yes[i]) < 0.3 and xes[i] > 100 :
            #print(txt)
            #ax.annotate(parser.get_specie_name('../data/train/', str(species_name[txt][0]) + '.xml'), (xes[i],yes[i]))
            ax.annotate(txt, (xes[i],yes[i]))
    plt.suptitle('False Positive', fontsize = 14)
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


def averages_if_less(seg, p, k, n):
    with open ('p_files/ROC_table_' + str(seg) + '_pc' + str(p) + '_k' + str(k) + '.p', 'rb') as f:
        ROC_table = pickle.load(f)
    with open ('p_files/species_stats.p', 'rb') as f:
        species_table = pickle.load(f)

    ROC_table = dict((k, ROC_table[k]) for k in ROC_table.keys() if species_table[k] > n)
    #print(len(ROC_table))
    #sys.exit(2)

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

    tp_sum = sum(float(item['tp']) for item in ROC_table.values())
    all_instances = sum(float(item['number']) for item in ROC_table.values())
    average = tp_sum / all_instances
    return recall_average, precision_average, f_average, average

def something(class_id):
    classified_as_1842 = gets_classified_as(float(class_id))

    classes = {}
    find_species_ex_func.find_species_ex(class_id, 2)
    for class_id, media_id, landmarks in classified_as_1842:
        if class_id in classes:
            classes[class_id] += 1
        else:
            classes[class_id] = 1


    sorted_list = sorted(classes.items(), key=itemgetter(1), reverse=True)
    sorted_list[:6]

#        for class_id, media_id, landmarks in classified_as_1842:
#            if class_id in
#
#
#        image = parser.get_image('../data/test/', media_id)
#        implot = plt.imshow(image)
#        xes = landmarks[::2]
#        yes = landmarks[1::2]
#        plt.plot(xes, yes, color='red')
#
#        plt.show()


    return sorted_list[:7]

def plot_fmeasure():
    with open ('p_files/ROC_table_' + str(seg) + '_pc' + str(p) + '_k' + str(k) + '.p', 'rb') as f:
        ROC_table = pickle.load(f)

    with open ('p_files/species_stats.p', 'rb') as f:
        species_table = pickle.load(f)

    with open('p_files/test_table_' + str(seg) + '_pc' + str(p) + '_k' + str(k) + '.p', 'rb') as f:
        test_table = pickle.load(f)

    with open ('p_files/image_table.p', 'rb') as f:
        image_table = pickle.load(f)

    #xes = [item['number'] for item in ROC_table.values()]
    #yes = [item['tp_rate'] for item in ROC_table.values()]
    #labels = ROC_table.keys()

    xes = []#[item['number'] for item in ROC_table.values()]
    yes = []#[item['fp'] for item in ROC_table.values()]
    label = []
    high_FP = []
    really_bad = []
    for specie in ROC_table:
        xes.append(species_table[specie])
        yes.append(ROC_table[specie]['f_measure'])
        label.append(specie)
        if float(ROC_table[specie]['f_measure']) == 0.0:
            really_bad.append((specie, species_table[specie], ROC_table[specie]['f_measure']))
    
    #xes = [item['number'] for item in ROC_table.values()]
    #yes = [item['f_measure'] for item in ROC_table.values()]
    #labels = ROC_table.keys()
    really_bad.sort(key=lambda tup: tup[1], reverse=True)
    print(really_bad)
    print(len(really_bad))

    #print(len(xes))
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.1)
    #plt.plot([0,max(xes)],[0.24845291479820627,0.24845291479820627], ls="--")
    ax.scatter(xes, yes, marker = '.')
    print('F measure')
    for i, txt in enumerate(label):
        if txt in interesting and xes[i] > 200 or float(yes[i]) > 0.42:
        #if xes[i] > 100 and float(yes[i]) < 0.3 or xes[i] > 200:
            print(txt)
            #ax.annotate(parser.get_specie_name('../data/train/', str(species_name[txt][0]) + '.xml'), (xes[i],yes[i]))
            ax.annotate(txt, (xes[i],yes[i]))
    #axes = plt.gca()
    #plt.suptitle('F measure', fontsize = 14)
    ax.set_xlabel('Number of Instances')
    ax.set_ylabel('F-measure')
    minor_ticks = np.arange(0, 350, 10)
    ax.set_xticks(minor_ticks, minor=True)
    #plt.savefig('plots/fmeasure_' + str(seg) + '.png')   # save the figure to file
    #plt.close()
    plt.show()




#def in_familie(liste):
#    with open('')


def make_printable(a_dict):
    print_string = ''
    for key, value in a_dict:
        print_string += key + ' : ' + '\\textit{' + str(value) + '} & '

    print_string += '\\\\'
    print(print_string)

with open ('p_files/species_stats.p', 'rb') as f:
    species_table = pickle.load(f)



interesting = ['30249', '3958', '1842', '3288','329', '5602','14872','4379','3956','1973','7305', '4109']
# '54', '2689', '6367'

#fór_3958 = {'4838': 3, '8631': 2, '30087': 1, '4719': 1, '3958': 18, '4379': 4, '3956': 7, '14900': 1, '1842': 8, '3955': 3, '326': 1, '5156': 1, '3798': 3, '30728': 1, '4763': 1, '2648': 1, '1837': 9, '3750': 5, '1973': 1}

#for_3958 = {'3958': 18, '4379': 4, '3956': 7, '1842': 8, '1837': 9, '3750': 5}
#for_3956 = [('3956', 17), ('3958', 6), ('5474', 4), ('4379', 4), ('1842', 4), ('5537', 2), ('1837', 2)]

#find_species_ex_func.find_species_ex(30040, 2, 'imt_')

#plot_PR()
#plot_fmeasure()
plot_Recall()
#plot_FP()
#plot_Precision_inv()
#plot_Precision()

#from operator import itemgetter
#something = something(6367)
#print(something)
#
#make_printable(something)


#sorted_list = sorted(species_table.items(), key=itemgetter(1))
#print(sorted_list[-6:])

#for i in ['4109', '2689', '7305', '1973']: #
#    find_species_ex_func.find_species_ex(i, 4, 'otsu_FP_')


#print(gets_classified_as(float('3958')))


#i = 0
#best_i = 0
#best_accuracy = 0
#while i < 223:
#    if averages_if_less(seg, p, k, i)[3] > best_accuracy:
#        best_accuracy = averages_if_less(seg, p, k, i)[3]
#        best_i = i
#        print('new_best: ', best_accuracy)
#        print('i: ', best_i)
#    i += 1
#print('i: ', best_i)
#print(averages_if_less(seg, p, k, best_i))
