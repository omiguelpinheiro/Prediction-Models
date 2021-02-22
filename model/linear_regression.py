import numpy as np

"""

Linear Regression Model: Uses examples features to fit the line that
gives us the lower loss for the examples. You can use several ways
to find this line, they are Batch Gradient Descent, Mini Batch
Gradient Descent, Stochastic Gradient Descent and finally using the
normal equation to directly jump to the local minima.

Math: Your prediction h(x) will be the weighted sum of each feature. 

h(x) = transpose(x) * w

Learn more: https://cutt.ly/JkK8z4D
Least Squares: https://cutt.ly/HkK8E9c at section 4.4

"""


class LinearRegression:
    def __init__(self):
        self.w0 = None

    def train(self, optimizer, epochs):
        self.w0 = optimizer.fit(optimizer.x_train,
                                optimizer.y_train,
                                epochs)

    def predict(self, x):
        predictions = x.dot(self.w0)
        return predictions

    def score(self, x_test, y_test, metric):
        predictions = x_test.dot(self.w0)
        score = metric.score(predictions, y_test)
        return score
