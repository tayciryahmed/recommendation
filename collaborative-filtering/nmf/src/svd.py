from sklearn.decomposition import TruncatedSVD
import numpy as np
from numpy import genfromtxt
import codecs
from numpy import linalg as LA
import config
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

class svd():
    def __init__(self):
        pass

    def svd(self, train_df):

        R = csr_matrix((train_df['rating'],
                  (train_df['new_user'], train_df['new_item']))).toarray()

        model = TruncatedSVD(n_components=config.K, random_state=42, \
                        tol=0.001, max_iter=config.steps)

        W = model.fit_transform(R)
        H = model.components_

        return u,s,v


    def score(self, train_df, test_df):
        u,s,v = self.svd(train_df)

        X = np.dot(np.dot(u, s), v)

        e, k = (0, 0)
        for i, j, z in zip(test_df['new_user'], test_df['new_item'], test_df['rating']):
            if i < len(X) and j < X.shape[1]:
                e += np.power(z - X[i,j], 2)
                k += 1.0

        return np.sqrt(float(e) / k)
