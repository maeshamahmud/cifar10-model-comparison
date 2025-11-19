import numpy as np

class ManualGaussianNB:
    #training for each class the mean, variance of each feature and the prior probabilty of that class
    def fit(self, X, y):
        n_samples, n_features = X.shape #5000,50

        self.classes = np.unique(y) #10 classes
        n_classes = len(self.classes)

        self.mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self.var = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]  

            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0) + 1e-6  # add small epsilon to avoid zeros

            #prior probability
            self.priors[idx] = X_c.shape[0] / float(n_samples)

    def _gaussian_log_likelihood(self, class_idx, x):
        mean = self.mean[class_idx]  
        var = self.var[class_idx]    

        # Formula: log N(x; mean, var) summed over all features
        log_likelihood = -0.5 * np.sum(
            np.log(2.0 * np.pi * var) + ((x - mean) ** 2) / var
        )

        return log_likelihood

    def _predict_one(self, x):
        log_posteriors = []

        for idx, c in enumerate(self.classes):
            log_prior = np.log(self.priors[idx])
            log_likelihood = self._gaussian_log_likelihood(idx, x)
            log_posterior = log_prior + log_likelihood
            log_posteriors.append(log_posterior)

        best_class_index = np.argmax(log_posteriors)
        return self.classes[best_class_index]

    def predict(self, X):
        predictions = [self._predict_one(x) for x in X]
        return np.array(predictions)

