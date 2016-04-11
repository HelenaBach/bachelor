### Statistics and evaluation of the different modules
### and functionalities of the plant project
import random
# When using as a script 
import knn 
import sys

# list of the methods 
methods_to_show = []

### main (module) ###

### parser ###

### segmentation ###

### asm ###

### pca ###

### knn ###

# show accuracy of classifier on the testset using cross validation with the given amount of folds
## arguments : the classifier to test, the list of the testset, number of fold for k fold cross validation
## return    : the accuracy of the classifier 
def knn_show_accuracy(classifier, test_list, fold):

	# shuffle the list
	random.shuffle(test_list)
	# partition the list in fold equally sized lists
	validation = [test_list[i::fold] for i in range(fold)]

	# For each of the 3 fold
	for index in range(fold):

	    # list() is used to ensure a deep copy 
	    val_list = list(validation_list)

	    # one of the lists should be used as test set
	    test = val_list.pop(index)
	    print(len(validation_list))

	    # merge the sublists to have one training set
	    train = [item for sublist in val_list for item in sublist]

	    # train classifier
	    clf = knn.train(train)

	    # measure accuracy on test data
	    for sample in test:
	    	# classify sample 
	    	knn.classify(clf, sample)

		print("Accuracy: {0}".format(accuracy))


### print statistics ###
## test if any flags are set or any methods has been called ##

# list of all flags
flags = [flag[1:] for flag in sys.argv if flag.startswith('-')]

# list of all stats to be printed
stats_to_show = flags + methods_to_show 
# 
for stat in stats_to_show:
	# show accuracy of knn 
	if stat == '-knn':
		# accuray should be a list where first element is the accuracy of the knn
		print("Accuracy: {0}".format(accuracy))
