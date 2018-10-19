import numpy
import matplotlib.pyplot as plt
from copy import deepcopy
import random
import math
import scipy
from scipy.stats import beta
from fractions import Fraction
import operator
import time
from matplotlib.patches import Polygon
import statistics
from decimal import *
from dirichlet import dirichlet
from scipy.special import gammaln
from gaussian import gaussian

#############################################################################################
####PURE BAYESIAN INFERENCE WITH GAUSSIAN CONJUGATE PRIOR####################################
#############################################################################################

class Bayesian_Inference_Gaussian(object):
	def __init__(self, prior, data, known_variance):
		self._prior = prior
		self._datasize = len(data)
		self._data = data
		self._posterior = prior
		self._known_variance = known_variance
		self._update_model(data)

	def _update_model(self, data):
		self._data = data
		self._datasize = len(data)
		self._update_posterior()
		return


	def _update_posterior(self):
		#################################################GENERAL GAUSSIAN BAYESIAN INFERENCE################################################################
		self._posterior = gaussian((1.0 / (1.0 / self._prior._variance + self._datasize / self._known_variance) 
			* (self._prior._mean / self._prior._variance + sum(self._data) / self._known_variance)),
			1.0 / (1.0 / self._prior._variance + self._datasize / self._known_variance))

		##########################################################STANDARD BAYESIAN INFERENCE###############################################################
		self._posterior = gaussian(math.floor(sum(self._data)) * 1.0 / (1.0 + self._datasize), 1.0 / (1.0 + self._datasize))

		print "POSTERIOR: " + str((self._posterior._mean, self._posterior._variance))


###############################################################################################################################################
#######################DIFFERENTIALLY PRIVATE BAYESIAN INFERENCE WITH GAUSSIAN CONJUGATE PRIOR#################################################
###############################################################################################################################################


class DP_Bayesian_Inference_Gaussian(object):
	def __init__(self, Bayesian_Inference_Gaussian, epsilon, delta):
		self._infer_model = Bayesian_Inference_Gaussian
		self._global_sensitivity_expmech = 0.0
		self._candidates = []
		self._probabilities = []
		self._epsilon = epsilon
		self._delta = delta
		self._smooth_sensitivity_expmech = 0.0
		self._private_posterior_exp_global = gaussian(0,1.0)
		self._private_posterior_exp_SS = gaussian(0,1.0)
		self._l1_sensitivity = 1.0/(1.0 + self._infer_model._datasize)

	def _update_model_setting(self):
		self._set_candidates()
		self._set_global_sensitivity()
		self._set_exponential_mechanism_prob()
		self._l1_sensitivity = 1.0/(1.0 + self._infer_model._datasize)



#####################################ENUMERATE ALL POSSIBLE CANDIDATES#######################################################################################
	def _set_candidates(self):
		n = self._infer_model._datasize
		self._candidates = []
		for r in range(n):
			self._candidates.append(gaussian((r/ (1.0 + n)), self._infer_model._posterior._variance))
		return


#####################################THE LOCAL SENSITIVITY####################################################################

	def _local_sensitivity(self,x):
		adjacents = x._adjacent(self._infer_model._datasize)
		return max([max([abs((x - r) - (y - r)) for y in adjacents]) for r in self._candidates])


#####################################THE EXPONENTIAL MECHANISM BASED ON GLOBAL SENSITIVITY####################################################################

	def _set_global_sensitivity(self):
		n = self._infer_model._datasize
		self._global_sensitivity_expmech = max([self._local_sensitivity(x) for x in self._candidates])
		print "GLOBAL SENSITIVITY: " + str(self._global_sensitivity_expmech)
#####################################THE EXPONENTIAL MECHANISM BASED ON GLOBAL SENSITIVITY####################################################################

	def _set_exponential_mechanism_prob(self):
		probabilities = []
		for r in self._candidates:
			probabilities.append(math.exp(- self._epsilon * (self._infer_model._posterior - r) / (2 * self._global_sensitivity_expmech)))

		self._probabilities = numpy.array(probabilities)/sum(probabilities)		
		return

	def _exponential_mechanism(self):
		r = numpy.random.choice(self._candidates, p=self._probabilities)
		self._private_posterior_exp_global = gaussian(r._mean, r._variance)
		return 	self._private_posterior_exp_global

####################################THE BASELINE APPROACH -- LAPLACE MECHANISM####################################################################

	def _laplace_mechanism(self):
		return 	gaussian(self._infer_model._posterior._mean + numpy.random.laplace(0, self._l1_sensitivity/self._epsilon), self._infer_model._posterior._variance)


#########################THE EXPONENTIAL MECHANISM BASED ON SMOOTH SENSITIVITY####################################################################


	def _set_smooth_sensitivity(self):

		self._smooth_sensitivity_expmech = max()
		return

	def _exponential_mechanism_smooth_sensitivity(self):
		probabilities = []
		for r in self._candidates:
			probabilities.append(math.exp(- self._epsilon * (self._infer_model._posterior - r) / self._global_sensitivity_expmech))

		probabilities = numpy.array(probabilities)/sum(probabilities)

		r = numpy.random.choice(self._candidates, p=probabilities)
		self._private_posterior_exp_SS = gaussian(r._mean, r._variance)

		return 	self._private_posterior_exp_SS


















