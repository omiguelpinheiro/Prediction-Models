"""

"""

class R2Score:
	def score(self, y_pred, y_true):
		numerator = ((y_true - y_pred) ** 2).sum()
		denominator = ((y_true - y_true.mean()) ** 2).sum()
		r2_score = 1 - numerator / denominator
		return r2_score
