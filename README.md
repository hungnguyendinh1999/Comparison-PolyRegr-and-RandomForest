# Comparison-PolyRegr-and-RandomForest

# 1. Provide a summary of the goal of your project. If you are replicating part of a paper, include a link to the paper here.

Our inspiration for this project comes from Section 8 experiments (Cheng et al. 2019) which attempt to prove that polynomial regression is equivalent or even better than neural network in several problems.

For this project, we will train two different models, one with Polynomial regression (PR), and the other, with Random forests (RF) theory (Louppe 2014) to compare the performance between them on a popular regression problem - NYC Taxi Trip Duration (Section 8.5, Cheng et al. 2019). Additionally, we wish to explore different evaluation metrics such as MAPE and RMSLE in regression problems (see part 5). Our initial goal is an MAPE of 580 and 591 for PR of degree 1 and 2 respectively. Then we’ll use that as a benchmark to compare with the performance of RF. Though, we will also experiment with higher polynomial orders to find the balance between model complexity and overfitting.

# 2. What dataset will you use?

The “New York City Taxi Trip Duration” dataset comes from a Kaggle competition to predict the duration of a . Here is the link https://www.kaggle.com/c/nyc-taxi-trip-duration.

## File descriptions:

- train.csv - the training set (contains 1,458,644 trip records)

- test.csv - the testing set (contains 625,134 trip records)

## Data fields:

- id - a unique identifier for each trip

- vendor_id - a code indicating the provider associated with the trip record

- pickup_datetime - date and time when the meter was engaged

- dropoff_datetime - date and time when the meter was disengaged

- passenger_count - the number of passengers in the vehicle (driver entered value)

- pickup_longitude - the longitude where the meter was engaged

- pickup_latitude - the latitude where the meter was engaged

- dropoff_longitude - the longitude where the meter was disengaged

- dropoff_latitude - the latitude where the meter was disengaged

- store_and_fwd_flag - This flag indicates whether the trip record was held in vehicle memory before sending to the vendor because the vehicle did not have a connection to the server - Y=store and forward; N=not a store and forward trip

- trip_duration - duration of the trip in seconds 3. What models will you train? You need to have more than one.

One set of models will be trained with Polynomial regression, and one will be trained with Random Forests algorithm. The PR approach will have at least 1st and 2nd degree models. Though, we will also experiment with higher polynomial orders.

# 4. What do you think will be most difficult about training the models?

There are two parts that might pose as problematic at first.

## Preprocessing steps:

- Handling outliers and missing values.

- Dealing with temporal and geospatial features and effectively encoding categorical variables.

## Model specific problems:

- Balancing model complexity and generality. Capturing the non-linear relationships and complex interactions between features (especially for PR) without succumbing to overfitting.

- RF training can be computationally demanding. While RF is robust against overfitting compared to individual decision trees, it can still occur, especially with noisy or highly correlated features while consuming a lot of the training resources.

## Other problems:

- Due to the team's lack of prior experience with Random Forest, additional time is needed to fully comprehend the method.

# 5. How will you evaluate the models?

In assessing the performance of our models, we will employ three metrics: Mean Squared Error (MSE), a standard measure in regression analysis from class, Mean Absolute Percentage Error (MAPE), the measurement used in the experiment in Table 9 (Cheng et al. 2019), and Root Mean Squared Logarithmic Error (RMSLE), suggested by the Kaggle community. This approach allows us to assess our models' performance using both traditional and newer evaluation criteria, providing a more comprehensive understanding of their predictive accuracy.

# Citations:

1. Cheng, X., Khomtchouk, B., Matloff, N. S., & Mohanty, P. (2019, April 11). Polynomial regression as an alternative to neural nets [ArXiv]. https://arxiv.org/abs/1806.06850

2. Louppe, Gilles. (2014). Understanding Random Forests: From Theory to Practice [ArXiv]. https://arxiv.org/pdf/1407.7502.pdf
