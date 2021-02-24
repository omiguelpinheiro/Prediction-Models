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
		sqr = ((y_true - y_pred) ** 2).sum()
		sqt = ((y_true - y_true.mean()) ** 2).sum()
		r2_score = 1 - sqr / sqt
		return r2_score


class AdjustedR2Score:
	def score(self, m, n, y_pred, y_true):
		r2_score = R2Score.score(y_pred, y_true)
		adjusted_r2 = 1 - ((m - 1) / (m - n + 1)) * (1 - r2_score)
		return adjusted_r2
