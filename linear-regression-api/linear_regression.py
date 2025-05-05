import numpy as np

class LinearRegressionScratch:
    def __init__(self):
        self.theta = None

    def fit(self, X, y):
        """
        Fit using pseudo-inverse: θ = (XTX)^⁺ XTy
        """
        # Add bias term (column of 1s) , giving constant value when the all features are 0.
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        # Compute weights using pseudo-inverse , this ensures the calculations even if the matrix is invertible
        self.theta = np.linalg.pinv(X_b).dot(y)

    def predict(self, X):
        """
        Predict using: ŷ = Xθ
        """
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.theta)

    def mean_squared_error(self, y_true, y_pred):
        """
        Compute mean squared error between predictions and actual values
        """
        return np.mean((y_true - y_pred) ** 2)

    def r2_score(self, y_true, y_pred):
        """
        Compute R2 score: 1 - (SS_res / SS_tot)
        """
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
