import numpy as np
np.random.seed()


def train_test_split(data, train_size=0.75, shuffle=True, random_state=None):
	np.random.seed(random_state)
	if not isinstance(data, np.ndarray):
		raise TypeError(f"data {data} is not a numpy array")

	if not isinstance(train_size, float):
		raise TypeError(f"train_size parameter should be float instead of {type(train_size)}")

	if not 0 <= train_size <= 1:
		raise ValueError(f"train_size should be 0 <= train_size <= 1")

	if not isinstance(shuffle, bool):
		raise TypeError(f"shuffle parameter should be bool instead of {type(shuffle)}")

	m = data.shape[0]

	if shuffle:
		rng = np.random.default_rng()
		rng.shuffle(data)

	x_train = data[: int(m * train_size), : -1]
	y_train = data[: int(m * train_size), -1]

	x_test = data[int(m * train_size):, :-1]
	y_test = data[int(m * train_size):, -1]

	return x_train, y_train, x_test, y_test


class StandardScaler:
	def __init__(self, scale_mean=True, scale_std=True):
		self.scale_mean = scale_mean
		self.scale_std = scale_std
		self.mean = None
		self.std = None

	def fit(self, data):
		self.mean = np.mean(data)
		self.std = np.std(data)
		return f"Mean: {self.mean}\nStandard Deviation: {self.std}"

	def transform(self, data):
		data = data - self.mean
		data = data / self.std
		return data

	def fit_transform(self, data):
		self.fit(data)
		data = self.transform(data)
		return data

	def inverse_transform(self, data):
		data = data * self.std
		data = data + self.mean
		return data
