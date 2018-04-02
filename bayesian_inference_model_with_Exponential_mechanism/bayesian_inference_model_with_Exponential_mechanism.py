import numpy
import matplotlib.pyplot as plt
from copy import deepcopy
import random
import math
import scipy

def beta_function(alpha, beta):
	return 1.0 * math.gamma(alpha) * math.gamma(beta)/math.gamma(alpha + beta)

def Hellinger_Distance(beta1, beta2):
	return math.sqrt(1 - beta_function((beta1[0] + beta2[0])/2.0, (beta1[1] + beta2[1])/2.0) \
		/ math.sqrt(beta_function(beta1[0], beta1[1]) * beta_function(beta2[0], beta2[1])))

class GenObservedDataset(object):

	def __init__(self, alpha, beta, sample_size, epsilon):

		self._alpha = alpha
		self._beta = beta
		self._sample_size = sample_size
		self._epsilon = epsilon
		self._theta = numpy.random.beta(self._alpha, self._beta)
		self._observation = numpy.random.binomial(1, self._theta, self._sample_size)
		self._ones = numpy.count_nonzero(self._observation == 1)
		self._zeros = numpy.count_nonzero(self._observation == 0)
		self._laplaced_ones = numpy.count_nonzero(self._observation == 1) + numpy.random.laplace(0, 1.0/self._epsilon)
		self._laplaced_zeros = numpy.count_nonzero(self._observation == 0) + numpy.random.laplace(0, 1.0/self._epsilon)
		self._randomized_ones = self._ones
		self._randomized_zeros = self._zeros
		self._randomized_observation = deepcopy(self._observation)
		self._exponential_ones = self._ones
		self._exponential_zeros = self._zeros

	def _set_randomized_ones(self):
		self._randomized_ones = numpy.count_nonzero(self._randomized_observation == 1)

	def _set_randomized_zeros(self):
		self._randomized_zeros = numpy.count_nonzero(self._randomized_observation == 0)

	def _randomize_observation(self):
		for i in range(self._sample_size):
			if random.randint(0,1) == 1:
				self._randomized_observation[i] = random.randint(0,1)
			else:
				self._randomized_observation[i] = self._observation[i]
		self._set_randomized_ones()
		self._set_randomized_zeros()

	def _set_exponential(self):
		R = [i + 1 for i in range(self._sample_size - 1)]#[(j + 1, i + 1) for i in range(self._sample_size) for j in range(self._sample_size)]
		def utility(r):
			return Hellinger_Distance((self._alpha, self._beta), (r, self._sample_size - r))	
		scores = {}
		sum_score = 0.0
		delta = 1
		for r in R:
			scores[r] = math.exp(self._epsilon * utility(r)/(2 * delta))
			sum_score = sum_score + scores[r]
		outpro = random.random()
		self._exponential_ones = R[0]
		for r in R:
			if outpro < 0:
				return
			outpro = outpro - scores[r]/sum_score
			self._exponential_ones = r
			self._exponential_zeros = self._sample_size - r
		return


	def _get_theta(self):
		return self._theta

	def _get_observation(self):
		return self._observation

	def _get_ones(self):
		return self._ones

	def _get_zeros(self):
		return self._zeros

	def _get_laplace_ones(self):
		return self._laplaced_ones

	def _get_laplace_zeros(self):
		return self._laplaced_zeros

	def _get_randomized_ones(self):
		return self._randomized_ones

	def _get_randomized_zeros(self):
		return self._randomized_zeros

	def _get_exponential_ones(self):
		return self._exponential_ones

	def _get_exponential_zeros(self):
		return self._exponential_zeros

	def _show_theta(self):
		print "The theta generated from the prior distribution is: " + str(self._theta)

	def _show_laplaced(self):
		print "The noised one # with Laplace mechanism is: " + str(self._laplaced_ones)
		print "The noised zero # with Laplace mechanism is: " + str(self._laplaced_zeros)

	def _show_randomized(self):
		print "The randomized data set is: "
		print self._randomized_observation
		print "The randomized one # is: " + str(self._randomized_ones)
		print "The randomized zero # is: " + str(self._randomized_zeros)

	def _show_observation(self):
		print "The observed data set is: "
		print self._observation

	def _show_exponential(self):
		print "The output ones # under Exponential Mechanism is: " + str(self._get_exponential_ones())
		print "The output zeros # under Exponential Mechanism is: " + str(self._get_exponential_zeros())

	def _show_all(self):
		self._show_theta()
		self._show_observation()
		self._show_laplaced()
		self._show_randomized()
		self._show_exponential()



def draw_distribution(datas, names):
	plt.subplots(nrows=len(datas), ncols=1, figsize=(10,len(datas) * 2.5))
	plt.tight_layout()
	for i in range(len(datas)):
		datas[i] = [round(item * 10 * 2.0)/2.0/10 for item in datas[i]]
		X = list(set(datas[i]))
		X.sort()
		Y = []
		for x in X:
			Y.append(datas[i].count(x))
		plt.subplot(len(datas),1,i+1)
		plt.bar(range(len(Y)), Y, facecolor='g', tick_label = X)
		plt.title(names[i])
	plt.show()

	return

if __name__ == "__main__":
	# Tests the functioning of the module
	alpha = 4
	beta = 5
	sample_size = 10
	epsilon = 10
	Bayesian_Model = GenObservedDataset(alpha, beta, sample_size, epsilon)
	Bayesian_Model._randomize_observation()
	Bayesian_Model._set_exponential()
	Bayesian_Model._show_all()

	draw_distribution([list(numpy.random.beta(alpha + Bayesian_Model._get_ones(),\
	beta + Bayesian_Model._get_zeros(), 1000)), \
	list(numpy.random.beta(alpha + Bayesian_Model._get_randomized_ones(),\
	beta + Bayesian_Model._get_randomized_zeros(), 1000)),\
	list(numpy.random.beta(alpha + Bayesian_Model._get_laplace_ones(),\
	beta + Bayesian_Model._get_laplace_zeros(), 1000)),\
	list(numpy.random.beta(alpha + Bayesian_Model._get_exponential_ones(),\
	beta + Bayesian_Model._get_exponential_zeros(), 1000))],\
	[" Bayesian Inferred Posterior","Randomized Response Posterior",\
	"(" + str(epsilon) + ", 0) - DP Posterior Using Laplace Mechanism",\
	"Exponential Mechanism Posterior"])
	




