import numpy as np
from optimizer.gradient_descent import GradientDescent

"""
Stochastic Gradient Descent: Is a gradient based optimization method,
we are using only ONE example of the dataset every time to make ONE
update on the model weights. It's called "Stochastic" because the
gradient based on a single example is a stochastic approximation of 
the true gradient. Because of it's stochasticity, the path to the
local / global minima is not direct but goes zig-zags. Useful on
large datasets.

Math: Uses Gradient Descent (check gradient_descent.py) math and the
loss function will be just the error of ONE example.

w = w - dw * lr

w = model weights
lr = learning rate (size of step in the -dw vector direction)
dw = gradient of the loss function with respect to the weights w

"""


class StochasticGradientDescent(GradientDescent):
	def __init__(self, x_train, y_train, learning_rate, random_state=None):
		super().__init__(x_train, y_train, learning_rate, 1, random_state)
