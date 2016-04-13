import numpy as np
from sklearn.decomposition import PCA


X = np.array([[-1, -1, -1, -1], [-2, -1, -2, -4], [-3, -2, -2, -2], [1, 1, 4, 2], [2, 1, 3, 5], [3, 2, 1, 3]])
pca = PCA(n_components=0.95) #'mle' eller 0.952½
pca.fit(X)
print(pca.explained_variance_ratio_)