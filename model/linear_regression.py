import numpy as np

"""

Linear Regression Model: Uses examples features to fit the line that
gives us the lower loss for the examples. You can use several ways
to find this line, they are Batch Gradient Descent, Mini Batch
Gradient Descent, Stochastic Gradient Descent and finally using the
normal equation to directly jump to the local minima.

Math: Your prediction h(x) will be the weighted sum of each feature. 

h(x) = transpose(x) * w

TODO: Talk about how to solve the equation dJ / dw = 0

Learn more: https://cutt.ly/JkK8z4D
Least Squares: https://cutt.ly/HkK8E9c at section 4.4

"""


class LinearRegression:
    def __init__(self, fit_intercept=True):
        self.w0 = None
        self.fit_intercept = fit_intercept

    def train(self, optimizer, epochs):
        self.w0 = optimizer.fit(optimizer.x_train,
                                optimizer.y_train,
                                epochs,
                                self.fit_intercept)

    def predict(self, x):
        if self.fit_intercept:
            x = np.insert(x, 0, 1)
        predictions = x.dot(self.w0)
        return predictions

    def score(self, x_test, y_test, metric):
        if self.fit_intercept:
            x_test = np.insert(x_test, 0, np.ones(len(x_test)), 1)
        predictions = x_test.dot(self.w0)
        score = metric.score(predictions, y_test)
        return score

    def solve(self, x_train, y_train):
        if self.fit_intercept:
            x_train = np.insert(x_train, 0, np.ones(len(x_train)), 1)
        self.w0 = np.linalg.inv(x_train.transpose().dot(x_train)).dot(x_train.transpose()).dot(y_train)

    def get_weights(self):
        return self.w0
