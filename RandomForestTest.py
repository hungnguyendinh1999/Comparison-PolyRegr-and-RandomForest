import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from RandomForest import DTree, RandomForest

### SAMPLE DATA ###
X_train = np.array([[1.5, 0.8, 0.4],
                   [2.0, 1.0, 2.3],
                   [2.2, 1.2, 8.4],
                   [2.4, 1.5, 3.1],
                   [3.0, 2.0, 6.8],
                   [3.5, 2.5, 5.3],
                   [4.0, 3.0, 1.7],
                   [4.5, 3.5, 12.3],
                   [5.0, 4.0, 4.2],
                   [5.5, 4.5, 15.2]])
Y_train = np.array([2.3, 1.2, 2.54, 4.6, 1.2, 1.9, 2.7, 1.2, 2.5, 3.8])

X_unseen = np.array([[3.0, 4.2, 7.9], [7.2, 8.6, 10.5], [10.2, 12.3, 6.9]])

### DECISION TREE ###
dtree = DTree(5)
dtree.train(X_train, Y_train)
print("Our model prediction: " + str(dtree.predict(X_unseen)))

# Comparing to Sklearn
# Since our implementation doesn't have any randomness involved, we need to set a fixed random seed
dtree = DecisionTreeRegressor(max_depth=5, random_state=1)
dtree.fit(X_train, Y_train)
print("Sklearn model prediction: " + str(dtree.predict(X_unseen)))


### RANDOM FOREST ###
rf = RandomForest(num_tree=100, max_tree_depth=5, max_feature=2, seed=1)
rf.train(X_train, Y_train)
print("Our model prediction: " + str(rf.predict(X_unseen)))

# Comparing to Sklearn
rf = RandomForestRegressor(max_depth=5, random_state=1, criterion='absolute_error', max_features=2, n_estimators=100)
rf.fit(X_train, Y_train)
print("Sklearn model prediction: " + str(rf.predict(X_unseen)))