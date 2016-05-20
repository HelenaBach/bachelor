import numpy as np
from table import image_table 
import sys
def construct():
	training_data = []
	for image_struct in image_table:
		# To ensure same type for all elements in the list -> change to float
		knn_vector = np.append(image_struct['feature_vector'], float(image_struct['class_id']))
		training_data.append(knn_vector)

	return training_data

# return the squared euclidean distance of the two vectores
# since square root is monotonic, no need to find the euclidean distance
def squared_euclidean_distance(vector1, vector2, length):
	distance = 0
	for x in range(length):
		distance += (vector1[x] - vector2[x])**2
	return distance

def getNeighbors(trainingSet, testInstance, k):
	distances = []
	# the last element in the test instance is NOT the class id
	length = len(testInstance)
	# calculate the distance from each training instance to the test instance
	for x in range(len(trainingSet)):
		dist = squared_euclidean_distance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	# sort the distances -> shortest first
	distances.sort(key=lambda item:item[1])
	neighbors = [x for (x, y) in distances[:k]]

	return neighbors
 
def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		# last element is the class id
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.items(), key=lambda item:item[1], reverse=True)
	# the number of votes that this classification got
	#votes = sortedVotes[0][1]
	#sum(i[1] for i in sortedVotes)
	# the classification is found by:
	# sortedVotes[0][0] 

	# return the whole list of sorted votes
	return sortedVotes


def classify(training_data, new_shape, k):

	neighbors = getNeighbors(training_data, new_shape, k)
	# get a sorted list of the class id and the number of votes
	# result[0][0] should give the classification  
	result = getResponse(neighbors)

	return result


