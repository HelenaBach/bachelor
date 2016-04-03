### Statistics and evaluation of the different modules
### and functionalities of the plant project
import random

### main ###

### parser ###

### segmentation ###

### asm ###
print("Accuracy: {0}".format(accuracy))

### pca ###
print("Accuracy: {0}".format(accuracy))


### knn ###

# show accuracy of classifier on the testset using cross validation with the given amount of folds
## arguments : the classifier to test, the list of the testset, number of fold for k fold cross validation
## return    : the accuracy of the classifier 
def knn_show_accuracy(classifier, test, fold):

	# shuffle the list
	random.shuffle(test)
	# partition the list in fold equally sized lists
	validation = [test[i::fold] for i in range(fold)]

	for i in range(fold):

	print("Accuracy: {0}".format(accuracy))


### print on terminal ###

# test if any flag(s) are set
