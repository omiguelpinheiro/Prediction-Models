import numpy as np

from loss.mean_squared_error import MeanSquaredError

"""
Batch Gradient Descent: Is a gradient based optimization method,
the "Batch" means it uses all of the dataset examples to make one
update on the model weights. It is the most precise way to reach a
local / global minima but weights updates takes forever for really
large datasets.

Math: One iteration of the algorithm makes your new weights be the
last weights you had less a fraction of the gradient of the loss
function with respect to the weights. This means that you are
finding how much each of the weights in your weight array is
contributing to the loss of your model and changing them in a way
that will make them contribute less to the loss. Repeat until
convergence.

w = w - dw * lr

w = model weights
lr = learning rate (size of step in the -dw vector direction)
dw = gradient of the loss function with respect to the weights w

"""


class GradientDescent:
    def __init__(self, x_train, y_train, learning_rate, batch_size, random_state):
        self.x_train = x_train
        self.y_train = y_train
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        np.random.seed(random_state)

    def fit(self, x_train, y_train, epochs, fit_intercept):
        loss_function = MeanSquaredError()
        if fit_intercept:
            x_train = np.insert(x_train, 0, np.ones(len(x_train)), 1)
        w = np.random.rand(x_train.shape[1]) / 1000
        for epoch in range(epochs):
            for batch in range(0, len(x_train), self.batch_size):
                x_batch = x_train[batch: batch + self.batch_size]
                y_batch = y_train[batch: batch + self.batch_size]
                loss_function.calculate_gradient(w, x_batch, y_batch)
                w = self.update_weights(w, loss_function.dw)
        return w

    def update_weights(self, w, dw):
        return w - self.learning_rate * dw

