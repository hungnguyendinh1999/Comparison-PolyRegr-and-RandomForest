import numpy as np


def _mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


class PolynomialRegression:
    """
    This class implements a Polynomial Regression model from scratch.
    """

    def __init__(self, degree, optimizer=None, cost_function=None):
        """
        Initializes the model with a specified polynomial degree.

        :param degree: the degree of polynomial
        :param optimizer: gradient descent
        :param cost_function: Custom cost function. If None, default to mean squared error.
        """

        self.degree = degree
        self.optimizer = optimizer
        self.coefficients = []
        self.cost_function = cost_function or _mean_absolute_percentage_error

    def fit(self, X, y):
        """
        Fit the model to the training data.

        :param X: (array-like) training input features
        :param y: (array-like) training labels
        """

        X_poly = self._create_polynomial_features(X)

        if self.optimizer is not None:
            n_features = X_poly.shape[1]
            initial_weights = np.random.randn(n_features)
            self.coefficients = self.optimizer.optimize(X_poly, y, initial_weights)
        else:
            # use normal equation method
            self.coefficients = np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T @ y

    def predict(self, X):
        X_poly = self._create_polynomial_features(X)
        return X_poly.dot(self.coefficients)

    def evaluate(self, X, y, cost_function=None):
        """
        Evaluate the model on the given data. Return the evaluation score.

        :param X: Input features
        :param y: Target labels
        :param cost_function: runnable cost function, or default one.
        :return: the evaluation score
        """

        y_pred = self.predict(X)
        cost_function = cost_function or self.cost_function
        return cost_function(y, y_pred)

    def _create_polynomial_features(self, X):
        """
        Create polynomial features from input data.
        """

        n_samples = X.shape[0]
        X_poly = np.ones((n_samples, 1))  # Include bias term
        for d in range(1, self.degree + 1):
            X_poly = np.hstack((X_poly, X ** d))
        return X_poly
