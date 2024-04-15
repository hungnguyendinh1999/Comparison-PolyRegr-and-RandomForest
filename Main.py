import csv
import matplotlib.pyplot as plt

import numpy as np


def read_csv(filename):
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)

    return data


# Compute the correlation matrix
num_features = ['pickup_datetime', 'dropoff_datetime', 'passenger_count', 'pickup_longitude',
                'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'trip_duration']
correlation_matrix = data[num_features].corr()

# Plot the correlation matrix heatmap
plt.figure(figsize=(10, 8))
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.title('Correlation Matrix Heatmap')
plt.xticks(np.arange(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
plt.yticks(np.arange(len(correlation_matrix.index)), correlation_matrix.index)
plt.show()
