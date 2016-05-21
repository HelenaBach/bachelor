import numpy as np

def fit(data, mean, dim):

	# subtract the mean of each dimension from
	# produces data set whose mean is zero

	# mean is an array of size # of landmarks * 2
	adjusted_data = data - mean


	# find eigenvectors and eigenvalues of covariance matrix of data
	U, k, V = np.linalg.svd(adjusted_data)
	#K = np.diag(k) -> k is a vector with the values of the diagonal

	# Columns of V are orthonormal eigenvectores of the covariance matrix
	eigenvector_matrix = V

	eigen_pair = []
	# for each eigenvector, find eigenvalue
	for i in range(len(eigenvector_matrix)):
		# eigenvalue is lambda_i = K_ii^2 / N, N = len(K) => K = N x D matrix
		eigenvalue = k[i]**2/len(k)
		eigen_pair.append((eigenvalue, eigenvector_matrix[i]))

	# sort according to the eigenvalue (in place)
#	eigen_pair.sort(key=lambda tup: tup[0], reverse=True)

	# all variance is explained by sum of eigenvalues
	total_variance = sum(i for i, j in eigen_pair)
	if dim < 1:
		# if dim is percentage:
		variance = 0
		principal_axis = []
		component_variance = []
		for value, vector in eigen_pair:
			variance += value / total_variance
			# eigenvectors as rows
			principal_axis.append(vector)
			# corresponding eigenvalue and variance
			component_variance.append((value, value / total_variance))
			if dim <= variance:
				break
	else:
		# if dim is a number
		# get the first dim eigenvectors
		# eigenvectors as rows
		principal_axis = [y for (x, y) in eigen_pair[:dim]]
		component_variance = [(x, x/total_variance) for (x, y) in eigen_pair[:dim]]

	return (principal_axis, component_variance)
