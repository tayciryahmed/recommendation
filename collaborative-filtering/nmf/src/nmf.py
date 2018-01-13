from sklearn.decomposition import NMF
import numpy as np
from numpy import genfromtxt
import codecs
from numpy import linalg as LA
import config
from scipy.sparse import csr_matrix

class nmf():
    def __init__(self):
        pass

    def create_model(self, train_df):

        R = csr_matrix((train_df['rating'],
                  (train_df['new_user'], train_df['new_item']))).toarray()

        model = NMF(n_components=config.K, init=config.init, random_state=42, \
                        tol=config.tol, max_iter=config.steps, alpha=config.beta)

        return model , R


    def score(self, train_df, test_df):
        self.model, self.R = self.create_model(train_df)

        P = self.model.fit_transform(self.R)
        Q = self.model.components_

        print "reconstruction_err: ", self.model.reconstruction_err_  / (self.R.shape[0]*self.R.shape[1])

        e, k = (0, 0)
        for i, j, z in zip(test_df['new_user'], test_df['new_item'], test_df['rating']):
            if i < len(P) and j < Q.shape[1]:
                #e += np.power(z - np.dot(P[i,:],Q[:,j]), 2)
                e += np.absolute(z - np.dot(P[i,:],Q[:,j]))
                k += 1.0

        return np.sqrt(float(e) / k)
