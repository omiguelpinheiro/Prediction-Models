"""

R2: Range goes from -infinity to 1 and explains how good of
a aproximation it is versus a horizontal line.

Math: We compare the sum of the squared residuals of our model
and the sum of the squared residuals of the data mean and
subtract this from one.

R2 = 1 - sqr / sqt

sqr = ((y_true / y_pred) ** 2).sum() = Residuals from model
sqt = ((y_true / y_mean) ** 2).sum() = Residuals from mean

"""


class R2Score:
	def score(self, y_pred, y_true):
		numerator = ((y_true - y_pred) ** 2).sum()
		denominator = ((y_true - y_true.mean()) ** 2).sum()
		r2_score = 1 - numerator / denominator
		return r2_score
