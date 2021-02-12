import numpy as np
from optimizer.gradient_descent import GradientDescent

"""
Mini Batch Gradient Descent: Is a gradient based optimization method,
the "Mini Batch" means it uses a mini batch of examples of the dataset
at a time to make one update on the model weights. It is less precise
than Batch Gradient Descent but weight updates are faster so it can
improve convergence time.

Math: Uses Gradient Descent (check gradient_descent.py) math and the
loss function will be the sum of the mini batch examples errors.

w = w - dw * lr

w = model weights
lr = learning rate (size of step in the -dw vector direction)
dw = gradient of the loss function with respect to the weights w

"""


class MiniBatchGradientDescent(GradientDescent):
	def __init__(self, learning_rate, batch_size):
		super().__init__(learning_rate, batch_size)
