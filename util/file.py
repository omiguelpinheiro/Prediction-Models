import csv
import numpy as np


def read_csv(filename, indexed=True):
	with open(filename) as csv_file:
		dict_reader = csv.DictReader(csv_file)
		examples = list(dict_reader)
		m = len(examples)
		n = len(examples[0].keys())
		if indexed:
			df = np.zeros((m, n - 1))
		else:
			df = np.zeros((m, n))
		for m, example in enumerate(examples):
			for n, value in enumerate(example.values()):
				if indexed:
					if n != 0:
						df[m][n - 1] = value
				else:
					df[m][n] = value
		return df
