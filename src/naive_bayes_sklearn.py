from sklearn.naive_bayes import GaussianNB

def train_sklearn_gaussian_nb(X_train, y_train):
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model
