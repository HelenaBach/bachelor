import pickle
import sys
#import test_knn.py

segs = ['otsu', 'ims']
pes = [13,28,50,80]
kes = [3,5,7,9]

def averages(seg, p, k):
    with open ('p_files/ROC_table_' + str(seg) + '_pc' + str(p) + '_k' + str(k) + '.p', 'rb') as f:
        ROC_table = pickle.load(f)

    recall_sum = sum(float(item['tp_rate']) for item in ROC_table.values())
    recall_sum_percentage = recall_sum
    recall_average = recall_sum_percentage / len(ROC_table)

    #print('Average of recall for ', seg, 'with ', p, ' PCs and ', k, ' neighbours ', recall_average)

    precision_sum = sum(float(item['precision']) for item in ROC_table.values())
    precision_sum_percentage = precision_sum
    precision_average = precision_sum_percentage / len(ROC_table)

    #print('Average of precision for ', seg, 'with ', p, ' PCs and ', k, ' neighbours ', precision_average)

    f_sum = sum(float(item['f_measure']) for item in ROC_table.values())
    f_sum_percentage = f_sum
    f_average = f_sum_percentage / len(ROC_table)

    #print('Average of F-measure for ', seg, 'with ', p, ' PCs and ', k, ' neighbours ', f_average)
    return recall_average, precision_average, f_average

for seg in segs:
    print(seg)
    for p in pes:
        for k in kes:
            with open('p_files/test_table_' + seg + '_pc' + str(p) + '_k' + str(k) + '.p', 'rb') as f:
                test_table = pickle.load(f)

            correct = 0

            for img in test_table:
                if img['prediction'] == True:
                    correct += 1

            accuracy = "{0:.3f}".format((correct/float(len(test_table))) * 100.0)

#print('Accuracy of ' + seg + ' with ' + str(p) + ' PCs and k = ' + str(k), accuracy)

            recalls, precisions, fmeasures = averages(seg,p,k)
            recall = "{0:.3f}".format(recalls)
            precision = "{0:.3f}".format(precisions)
            f_measure = "{0:.3f}".format(fmeasures)

            print(p, '&', k, '&', recall,'&', precision, '&', f_measure, '&', accuracy)

#sys.exit(4)

#try:
#    seg = sys.argv[1]
#except:
#    print('otsu or ims')
#    sys.exit(3)
#try:
#    p = sys.argv[2]
#except:
#    print('number of PCs')
#    sys.exit(3)
#try:
#    k = sys.argv[3]
#except:
#    print('k = 3,5,7,9')
#    sys.exit(3)
