import numpy as np

"""

Linear Regression Model: Uses examples features to fit the line that
gives us the lower loss for the examples. You can use several ways
to find this line, they are Batch Gradient Descent, Mini Batch
Gradient Descent, Stochastic Gradient Descent and finally using the
normal equation to directly jump to the local minima

Math: Your prediction h(x) will be the weighted sum of each feature

h(x) = transpose(x) * w

Linear Regression Models are unique in the way that you don't actually
need to use gradient descent to find local minima because the loss
function is always convex, meaning that there is only one local minima
that is the global minima

Math: We want to minimize the square of the eucledian distance between
our examples predictions and their true values

||Ax - b|| ** 2 = (Ax - b)T(Ax - b) = xTATAx - 2bTAx + bTb

Taking the gradient with respect to x

grad(xTATAx) - grad(2bTAx) + grad(bTb) = 2ATAx - 2ATb

Solving the gradient last equation for 0

x = ((ATA) ** -1)ATb 

Learn more: https://cutt.ly/JkK8z4D
Matrix Properties, Matrix Calculus and Least Squares: https://cutt.ly/HkK8E9c

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
