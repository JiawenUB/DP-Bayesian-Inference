import numpy
import matplotlib.pyplot as plt

class GenObservedDataset(object):



	def __init__(self, alpha, beta, sample_size):

		self._alpha = alpha
		self._beta = beta
		self._sample_size = sample_size
		self._theta = numpy.random.beta(self._alpha, self._beta)
		self._observation = numpy.random.binomial(1, self._theta, self._sample_size)

	def _get_theta(self):
		return self._theta
		"""Equalizes the number of 1 and -1 labels"""

	def _get_observation(self):
		return self._observation

	def _get_ones(self):
		return numpy.count_nonzero(self._observation == 1)

	def _get_zeros(self):
		return numpy.count_nonzero(self._observation == 0)



def observing(given_database, alpha, beta, sample_size, times = 1):
	posterior_thetas = []
	prior_thetas = []
	count = 0
	while count < times:
		observation = GenObservedDataset(alpha, beta, sample_size)
		prior_thetas.append(observation._get_theta())
		if (given_database._get_observation() == observation._get_observation()).all():
			posterior_thetas.append(observation._get_theta())
			count = count + 1
	return prior_thetas, posterior_thetas

def draw_distribution(datas, titles):
	plt.subplots(nrows=len(datas), ncols=1, figsize=(8,len(datas)*3))
	for i in range(len(datas)):
		datas[i] = [round(item,2) for item in datas[i]]#[round(item * 10 * 2.0)/2.0/10 for item in datas[i]]
		X = list(set(datas[i]))
		X.sort()
		Y = []
		for x in X:
			Y.append(datas[i].count(x))
		plt.subplot(len(datas),1,i+1)
		plt.bar(range(len(Y)), Y, facecolor='g', tick_label = X)
		plt.title(titles[i])
	plt.show()

	return

if __name__ == "__main__":
	# Tests the functioning of the module
	alpha = 4
	beta = 5
	sample_size = 5
	given_database = GenObservedDataset(alpha, beta, sample_size)
	prior_thetas, posterior_thetas = observing(given_database, alpha, beta, sample_size, 1000)
	print given_database._get_observation()

	draw_distribution([prior_thetas, posterior_thetas, list(numpy.random.beta(alpha \
		+ given_database._get_ones(), beta + given_database._get_zeros(), 1000))], \
		['prior','observed_posterior','calculated_posterior'])



