import numpy
import matplotlib.pyplot as plt

class GenObservedDataset(object):

	"""Generate a Random DataSet with binary class labels.

	All data points are randomly chosen from a normal distribution centered at the origin and having std deviation 1
	Data is generated as a 2-D matrix of values (rows) with a fixed number of attributes (columns).
	The last attribute of every value (row) is corresponds to the class label : 1 or -1.
	The generated Data is split into three sub-sets : train, test and valid.

	Attributes:
		train: The training data set feature vectors
		valid: The validation data set feature vectors
		test: The testing data set feature vectors
		train_labels: The training data set feature labels
		valid_labels: The validation data set feature labels
		test_labels: The test data set feature labels

	"""

	def __init__(self, alpha, beta, sample_size, epsilon):
		"""Specify the data set parameters
		Args:
		Raises:
			ValueError : if split_ratio is invalid.
		"""
		self._alpha = alpha
		self._beta = beta
		self._sample_size = sample_size
		self._theta = numpy.random.beta(self._alpha, self._beta)
		self._observation = numpy.random.binomial(1, self._theta, self._sample_size)
		self._ones = numpy.count_nonzero(self._observation == 1)
		self._zeros = numpy.count_nonzero(self._observation == 0)
		self._noised_ones = numpy.count_nonzero(self._observation == 1) + numpy.random.laplace(0, 1.0/epsilon)
		self._noised_zeros = numpy.count_nonzero(self._observation == 0) + numpy.random.laplace(0, 1.0/epsilon)

	def _get_theta(self):
		return self._theta
		"""Equalizes the number of 1 and -1 labels"""

	def _get_observation(self):
		return self._observation

	def _get_ones(self):
		return self._ones

	def _get_zeros(self):
		return self._zeros

	def _get_Laplace_ones(self):
		return self._noised_ones

	def _get_Laplace_zeros(self):
		return self._noised_zeros

	def _show_theta(self):
		print "The theta generated from the prior distribution is: " + str(self._theta)

	def _show_noised(self):
		print "The noised one # is: " + str(self._noised_ones)
		print "The noised zero # is: " + str(self._noised_zeros)

	def _show_observation(self):
		print "The observed data set is: "
		print self._observation

	def _show_all(self):
		self._show_observation()
		self._show_theta()
		self._show_noised()



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

def draw_distribution(datas, names):
	plt.subplots(nrows=len(datas), ncols=1, figsize=(8,len(datas)*3))
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
	sample_size = 5
	epsilon = 10
	Bayesian_Model = GenObservedDataset(alpha, beta, sample_size, epsilon)
	Bayesian_Model._show_all()

	draw_distribution([list(numpy.random.beta(alpha + Bayesian_Model._get_ones(),\
	beta + Bayesian_Model._get_zeros(), 1000)), \
	list(numpy.random.beta(alpha + Bayesian_Model._get_Laplace_ones(),\
	beta + Bayesian_Model._get_Laplace_zeros(), 1000))],\
	[" Bayesian Inferred Posterior","(" + str(epsilon) + ", 0) - DP Posterior Using Laplace Mechanism"])




