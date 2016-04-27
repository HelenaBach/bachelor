import numpy as np
#from sklearn.decomposition import PCA

#data = np.array([[-1, -1, -1, -1], [-2, -1, -2, -4], [-3, -2, -2, -2], [1, 1, 4, 2], [2, 1, 3, 5], [3, 2, 1, 3]])

def train_get_components(data, mean, dim):
#	pca = PCA(n_components=acc)#0.95) #'mle' eller 2
#	pca.fit(data)
#	return pca.components_
#
#	#print(pca.explained_variance_ratio_)
#	#print(pca.components_)
	
	# subtract the mean of each dimension from
	# produces data set whose mean is zero

	# mean is an array of size # of landmarks * 2 
	adjusted_data = data - mean

	# find eigenvectors and eigenvalues of covariance matrix of data
	U, K, V = np.linalg.svd(adjusted_data)

	# Columns of V are orthonormal eigenvectores of the covariance matrix
	eigenvector_matrix = V.transpose()

	eigen_pair = []
	# for each eigenvector, find eigenvalue
	for i in range(len(eigenvector_matrix)): # <- should maybe just be V?
		# eigenvalue is lambda_i = K_ii^2 / N, N = len(K) => K = N x D matrix
		eigenvalue = int(K[i][i]^2/len(K))
		eigen_pair[i] = (eigenvalue, eigenvector_matrix[i])

	# sort according to the eigenvalue (in place)
	eigen_pair.sort(key=lambda tup: tup[1])

	if dim < 1:
		# if dim is percentage:
		## all variance is explained by sum of eigenvalues
		total_variance = sum(i for i, j in eigen_pair)
		variance = 0
		feature_vector = []
		for value, vector in eigen_pair:
			variance += value / total_variance
			# eigenvectors as rows
			feature_vector.append(vector)
			if dim <= variance:
				break
	else:
		# if dim is a number
		# get the first dim eigenvectors
		# eigenvectors as rows
		feature_vector = [y for (x, y) in eigen_pair[:dim]]

	# maybe we want this to be done in ACM instead of in the PCA?
	# then simply return the dimensions and maybe the adjusted data
	#final_data = np.dot(feature_vector, adjusted_data)
	#return final_data
	return (feature_vector, adjusted_data)