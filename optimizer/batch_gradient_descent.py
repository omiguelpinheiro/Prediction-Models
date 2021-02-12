import numpy as np

from optimizer.gradient_descent import GradientDescent

"""
Batch Gradient Descent: Is a gradient based optimization method,
the "Batch" means it uses all of the dataset examples to make one
update on the model weights. It is the most precise way to reach a
local / global minima but weight updates takes forever for really
large datasets.

Math: Uses Gradient Descent (check gradient_descent.py) math and the
loss function will be the sum of all examples errors.

w = w - dw * lr

w = model weights
lr = learning rate (size of step in the -dw vector direction)
dw = gradient of the loss function with respect to the weights w

"""


class BatchGradientDescent(GradientDescent):
	def __init__(self, learning_rate):
		super().__init__(learning_rate, -1)
