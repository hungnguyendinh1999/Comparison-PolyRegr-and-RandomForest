import numpy as np


class GradientDescent:
    """
    Gradient Descent implementation made by {@author Hung Nguyen}
    """

    def __init__(self, cost_function, learning_rate=0.001, max_iter=1000, tol=1e-6):
        self.cost_function = cost_function
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol

    def optimize(self, X, y, initial_weights):
        """
        Optimize the model parameters using gradient descent.

        :param X: (array-like) the input features.
        :param y: (array-like) the target values.
        :param initial_weights: (array-like) the initial guess for model parameters.
        :return: (np-array) the optimized model parameters.
        """

        coefficients = np.array(initial_weights)
        prev_cost = float('inf')  # infinity
        n_samples = len(X)

        for _ in range(self.max_iter):
            # make predictions via dot product
            y_pred = np.dot(X, coefficients)

            cost = self.cost_function(y, y_pred)

            # check convergence
            if abs(prev_cost - cost) < self.tol:
                break

            prev_cost = cost
            # compute gradient and update weights
            gradient = np.dot(X.T, y_pred - y) / n_samples
            coefficients -= self.learning_rate * gradient

        return coefficients