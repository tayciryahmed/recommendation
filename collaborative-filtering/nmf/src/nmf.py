from sklearn.decomposition import NMF


class nmf():
    def __init__(self):
        pass

    def nmf(self, X_train):
        model = NMF(n_components=2, init='random', random_state=0, tol=0.1)
        W = model.fit_transform(X_train)
        H = model.components_
        return W, H

    def score(self, X_train, test_df):
        W, H = self.nmf(X_train)
        X = W.dot(H)
        e, k = (0, 0)
        for i, j, z in zip(test_df['new_user'], test_df['new_item'], test_df['rating']):
            if i < len(X) and j < X.shape[1]:
                e += abs(X[i, j] - z)                
                k += 1.0

        return float(e) / k
