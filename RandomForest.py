import numpy as np
import math

class Node:
    """
    Represent a node in the decision tree
    """
    def __init__(self, left=None, right=None, feature_index=None, threshold=None, value=None):
        self.left = left
        self.right = right
        self.feature_index = feature_index
        self.threshold = threshold
        self.value = value


class DTree:
    """
    Decision tree implemented from scratch, using Mean Absolute Percentage error
    """
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.model = None

    def train(self, X, Y):
        """
        Train this model with input feature data and the corresponding class.
        Required to run before being able to provide prediction.
        """
        self.model = self._grow(X, Y)

    def predict(self, X):
        """
        Make prediction on provided X feature data using the trained model
        """
        # Predicting every x examples' feature values with the calculated decision tree
        return np.array([self._predict(x, self.model) for x in X])

    def _grow(self, X, Y, depth=0):
        num_samples, num_features = X.shape
        num_classes = len(np.unique(Y))

        best_mape = math.inf
        best_index = None
        best_threshold = None
        best_sets = None

        # Base case, if there's only 1 class or has reach max depth
        if num_classes == 1 or depth == self.max_depth:
            return Node(value=np.mean(Y))

        # Loop through all the features
        for feature_index in range(num_features):
            # Find thresholds
            thresholds = np.unique(X[:, feature_index])

            # For each threshold, split the chosen feature set into half base on the value of chosen threshold.
            # Our goal is to find the best threshold to divide classes
            for threshold in thresholds:
                left_indices = np.where(X[:, feature_index] <= threshold)[0]
                right_indices = np.where(X[:, feature_index] > threshold)[0]

                # If one or the other side is empty, we don't consider it
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue

                # Perform MAPE on both left and right
                Y_left = Y[left_indices]
                Y_right = Y[right_indices]

                Y_pred_left = np.mean(Y_left)
                Y_pred_right = np.mean(Y_right)

                mape_left = self._mean_absolute_percentage_error(Y_left, Y_pred_left)
                mape_right = self._mean_absolute_percentage_error(Y_right, Y_pred_right)

                mape = (np.sum(left_indices) * mape_left + np.sum(right_indices) * mape_right) / num_samples

                # If that mape is better than the current best, we store it to use later
                if mape < best_mape:
                    best_mape = mape
                    best_index = feature_index
                    best_threshold = threshold
                    best_sets = (left_indices, right_indices)

        # If for any reason mape score hasn't been calculated
        if best_mape == math.inf:
            return Node(value=np.mean(Y))

        # Continue to grow recursively of both left and right side and append it to the current node
        # using the best sets that we have found above
        left = self._grow(X[best_sets[0]], Y[best_sets[0]], depth + 1)
        right = self._grow(X[best_sets[1]], Y[best_sets[1]], depth + 1)

        return Node(feature_index=best_index, threshold=best_threshold, left=left, right=right)

    def _mean_absolute_percentage_error(self, Y_actual, Y_pred):
        return np.mean(np.abs((Y_actual - Y_pred) / Y_actual)) * 100

    def _predict(self, x, node):
        # Return prediction value if it's available, aka. bottom of the tree
        if node.value is not None:
            return node.value

        # Else descend lower into the tree and return the final prediction all the way up
        if x[node.feature_index] <= node.threshold:
            return self._predict(x, node.left)
        else:
            return self._predict(x, node.right)


class RandomForest:
    """
    Random forest regressor implemented from scratch, using Mean Absolute Percentage Error
    """
    def __init__(self, num_tree, max_tree_depth, max_feature, seed):
        self.num_tree = num_tree
        self.trees = []
        self.max_tree_depth = max_tree_depth
        self.max_feature = max_feature
        self.seed = seed

    def train(self, X, Y):
        """
        Train this model with input feature data and the corresponding class.
        Required to run before being able to provide prediction.
        """
        np.random.seed(self.seed)

        for _ in range(self.num_tree):
            # Choosing random examples to train with, or bootstrapping with replacement
            sample_indices = np.random.choice(X.shape[0], size=X.shape[0], replace=True)

            # Choosing random features to train with without replacement
            feature_indices = np.random.choice(X.shape[1], size=self.max_feature, replace=False)

            # Compiling into a training data format
            X_training = X[sample_indices][:, feature_indices]
            Y_training = Y[sample_indices]

            dtree = DTree(self.max_tree_depth)
            dtree.train(X_training, Y_training)
            self.trees.append(dtree)

    def predict(self, X):
        """
        Make prediction on provided X feature data using the trained model
        """
        predictions = []

        # Calculate result for individual trees
        for tree in self.trees:
            predictions.append(tree.predict(X))

        # Averaging from all the predictions each trees made
        return np.mean(predictions, axis=0)

