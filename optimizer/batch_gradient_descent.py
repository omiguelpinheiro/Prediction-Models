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
	def __init__(self, x_train, y_train, learning_rate, random_state=None):
		super().__init__(x_train, y_train, learning_rate, len(x_train), random_state)
