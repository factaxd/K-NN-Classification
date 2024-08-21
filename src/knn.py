import numpy as np
from collections import Counter
import pandas as pd

class KNearestNeighbors:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        # Reset indices for y_train if it's a pandas Series
        if isinstance(y_train, pd.Series):
            self.y_train = y_train.reset_index(drop=True)
        else:
            self.y_train = y_train

    def predict(self, X_test):
        # Make predictions for each sample in X_test
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)

    def _predict(self, x):
        # Calculate Euclidean distances between the input x and all X_train samples
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        # Get the indices of the k-nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        
        # Fetch the labels of the k-nearest neighbors
        k_nearest_labels = [self.y_train[int(i)] for i in k_indices]
        
        # Return the most common label (mode) among the k-nearest neighbors
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
