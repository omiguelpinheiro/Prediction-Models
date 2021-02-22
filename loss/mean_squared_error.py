"""
Mean Squared Error (MSE): This loss is useful because it treats big
errors as more relevant than small errors.

Math: The loss is the mean of the sum of the squared differences 
between all of the predictions h(x) and their true values y. It's
common to multiply the MSE by 1 / 2 to make the derivation easier.
We can do this because what we are trying to do is minimize the
model error so it can go as low as possible and trying to do this
to the error and to half of the error is the same thing. We are
still trying to minimize the error so it can get to it's lowest. 

J(h(x), y) = (1 / m) * sum((h(x) - y) ** 2) * (1 / 2)

J = Mean Squared Error loss function
h(x) = the model predictions for the x example
m = number of examples
sum = sum over all the m examples

"""


class MeanSquaredError:
	def __init__(self):
		self.dw = None
		self.loss = None

	def calculate_gradient(self, w, x_train, y_train):
		m = len(x_train)
		predictions = x_train.dot(w)
		residuals = y_train - predictions
		self.dw = - x_train.transpose().dot(residuals) / m
		self.loss = (1 / 2 * m) * sum(residuals ** 2)
