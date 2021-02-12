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
    def __init__(self, x_train, y_train):
        self.w0 = np.random.rand(x_train.shape[1])
        self.x_train = x_train
        self.y_train = y_train

    def train(self, optimizer, epochs):
        self.w0 = optimizer.fit(self.w0,
                                self.x_train,
                                self.y_train,
                                epochs)

    def test(self, x_test, y_test):
        pass
