import numpy as np
from sklearn.decomposition import PCA

#data = np.array([[-1, -1, -1, -1], [-2, -1, -2, -4], [-3, -2, -2, -2], [1, 1, 4, 2], [2, 1, 3, 5], [3, 2, 1, 3]])

def train_get_components(data, acc):
	pca = PCA(n_components=acc)#0.95) #'mle' eller 2
	pca.fit(data)
	return pca.components_

	#print(pca.explained_variance_ratio_)
	#print(pca.components_)