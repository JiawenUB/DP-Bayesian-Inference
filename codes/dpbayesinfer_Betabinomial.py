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


def Hamming_Distance(c1, c2):
	temp = [abs(a - b) for a,b in zip(c1,c2)]
	return sum(temp)/2.0


class BayesInferwithDirPrior(object):
	def __init__(self, prior, sample_size, epsilon, delta):
		self._prior = prior
		self._sample_size = sample_size
		self._epsilon = epsilon
		self._delta = delta
		self._bias = numpy.random.dirichlet(self._prior._alphas)
		self._observation = numpy.random.multinomial(1, self._bias, self._sample_size)
		self._observation_counts = numpy.sum(self._observation, 0)
		self._posterior = dirichlet(self._observation_counts) + self._prior
		self._laplaced_posterior = self._posterior
		self._laplaced_zhang_posterior = self._posterior
		self._exponential_posterior = self._posterior
		self._SS_posterior = self._posterior
		self._candidate_scores = {}
		self._candidates = []
		self._LS_probabilities = []
		self._GS_probabilities = []
		self._SS_probabilities = []
		self._alpha_SS_probabilities = []
		self._LS_Candidates = {}
		self._GS = 0.0
		self._LS = 0.0
		self._SS = 0.0
		self._alpha_SS = 0.0
		self._keys = []
		self._accuracy = {}
		self._accuracy_mean = {}
	
	def _set_bias(self, bias):
		self._bias = bias
		self._update_observation()

	def _set_observation(self,observation):
		self._observation_counts = observation
		self._posterior = dirichlet(observation) + self._prior

	def _get_accuracy_bound(self, c, delta):
		nominator = 0.0
		for r in self._candidates:
			if -self._candidate_scores[r] > c:
				nominator += math.exp(self._candidate_scores[r] * self._epsilon/ (delta))

		return nominator

	def _get_approximation_accuracy_bound(self, c, delta):
		return len(self._candidates) * math.exp(- self._epsilon * c/ delta)


	def _update_observation(self):
		self._observation = numpy.random.multinomial(1, self._bias, self._sample_size)
		self._posterior = dirichlet(self._observation_counts) + self._prior

	def _set_candidate_scores(self):
		start = time.clock()
		self._set_candidates([], numpy.sum(self._observation))

		for r in self._candidates:
			self._candidate_scores[r] = -(self._posterior - r)

	def _set_candidates(self, current, rest):
		if len(current) == len(self._prior._alphas) - 1:
			current.append(rest)
			self._candidates.append(dirichlet(deepcopy(current)) + self._prior)
			current.pop()
			return
		for i in range(0, rest + 1):
			current.append(i)
			self._set_candidates(current, rest - i)
			current.pop()

	def _set_local_sensitivities(self):
		for r in self._candidates:
			self._LS_Candidates[r] = r._hellinger_sensitivity()

###################################################################################################################################
#####SETTING UP THE MECHANISMS BEFORE DOING EXPERIMENTS
###################################################################################################################################	


	###################################################################################################################################
	#####SETTING UP THE BASELINE LAPLACE MECHANISM
	###################################################################################################################################	

	def _set_up_baseline_lap_mech(self):
		self._keys.append("Baseline LaplaceMech")
		self._accuracy["Baseline LaplaceMech"]=[]
		self._accuracy_mean["Baseline LaplaceMech"]=[]

	###################################################################################################################################
	#####SETTING UP THE IMRPOVED LAPLACE MECHANISM
	###################################################################################################################################	
	def _set_up_improved_lap_mech(self):
		self._keys.append("Improved LaplaceMech")
		self._accuracy["Improved LaplaceMech"]=[]
		self._accuracy_mean["Improved LaplaceMech"]=[]


	###################################################################################################################################
	#####SETTING UP THE EMPONENTIAL MECHANISM WITH THE SMOOTH SENSITIVITY
	###################################################################################################################################	
	def _set_up_exp_mech_with_SS(self):

		###################################################################################################################################
		#INITIALIZE THE LISTS FOR EXPERIMENTS
		###################################################################################################################################	
		self._keys.append("ExpoMech of SS")
		self._accuracy["ExpoMech of SS"] = []
		self._accuracy_mean["ExpoMech of SS"] = []

		#############################################################################################################
		#CALCULATING THE SENSITIVITY
		###################################################################################################################################
		t0 = time.time()
		start = time.clock()
		beta = 0.01 # math.log(1 - self._epsilon / (2.0 * math.log(self._delta / (2.0 * (self._sample_size)))))
		self._SS = max(self._LS, max([self._LS_Candidates[r] * math.exp(- beta * Hamming_Distance(self._observation_counts, [r._alphas[i] - self._prior._alphas[i] for i in range(self._prior._size)])) for r in self._candidates]))
		t1 = time.time()
		print("smooth sensitivity"+str(t1 - t0)), self._SS


		###################################################################################################################################
		#CALCULATING THE OUTPUTTING PROBABILITIES
		###################################################################################################################################
		nomalizer = 0.0
		for r in self._candidates:
			temp = math.exp(self._epsilon * self._candidate_scores[r]/(4 * self._SS))
			self._SS_probabilities.append(temp)
			nomalizer += temp

		for i in range(len(self._SS_probabilities)):
			self._SS_probabilities[i] = self._SS_probabilities[i]/nomalizer
		return nomalizer



	###################################################################################################################################
	#####SETTING UP THE EXPONENTIAL MECHANISM WITH THE ALPHA SMOOTH SENSITIVITY
	###################################################################################################################################	

	def _set_up_exp_mech_with_alpha_SS(self):

		###################################################################################################################################
		#INITIALIZE THE LISTS FOR EXPERIMENTS
		###################################################################################################################################		
		self._keys.append("ExpoMech of alpha SS")
		self._accuracy["ExpoMech of alpha SS"] = []
		self._accuracy_mean["ExpoMech of alpha SS"] = []

		###################################################################################################################################
		#CALCULATING THE SENSITIVITY
		###################################################################################################################################
		t0 = time.time()
		start = time.clock()
		gamma = 1.0
		self._alpha_SS = max([(1.0 / (1.0/self._LS_Candidates[r] + gamma * Hamming_Distance(self._observation_counts, [r._alphas[i] - self._prior._alphas[i] for i in range(self._prior._size)]))) for r in self._candidates])
		t1 = time.time()
		# print ("alpha smooth sensitivity"+str(t1 - t0)), self._alpha_SS


		###################################################################################################################################
		#CALCULATING THE SENSITIVITY
		###################################################################################################################################
		x = self._observation_counts
		nomalizer = 0.0
		for r in self._candidates:
			temp = math.exp(self._epsilon * self._candidate_scores[r]/(2 * self._alpha_SS))
			self._alpha_SS_probabilities.append(temp)
			nomalizer += temp

		for i in range(len(self._alpha_SS_probabilities)):
			r = self._candidates[i]
			self._alpha_SS_probabilities[i] = self._alpha_SS_probabilities[i]/nomalizer
		return nomalizer



	###################################################################################################################################
	#####SETTING UP THE EXPONENTIAL MECHANISM WITH THE LOCAL SENSITIVITY
	###################################################################################################################################	
	def _set_up_exp_mech_with_LS(self):

		self._accuracy["Expomech of LS"] = []
		self._keys.append("Expomech of LS")

		
		self._LS = self._posterior._hellinger_sensitivity()

		
		nomalizer = 0.0
		for r in self._candidates:
			temp = math.exp(self._epsilon * self._candidate_scores[r]/(2 * self._LS))
			self._LS_probabilities.append(temp)
			nomalizer += temp

		for i in range(len(self._LS_probabilities)):
			self._LS_probabilities[i] = self._LS_probabilities[i]/nomalizer

		return nomalizer




	###################################################################################################################################
	#####SETTING UP THE STANDARD EXPONENTIAL MECHANISM
	###################################################################################################################################	
	def _set_up_exp_mech_with_GS(self):


		self._accuracy["ExpoMech of GS"] = []
		self._keys.append("ExpoMech of GS")

		self._GS = math.sqrt(1 - math.pi/4)

		nomalizer = 0.0
		for r in self._candidates:
			temp = math.exp(self._epsilon * self._candidate_scores[r]/(2 * self._GS))
			self._GS_probabilities.append(temp)
			nomalizer += temp

		for i in range(len(self._GS_probabilities)):
			self._GS_probabilities[i] = self._GS_probabilities[i]/nomalizer

		return nomalizer


	
########################################################################################################################################
######EXPERMENTING FUNCTION, I.E., SAMPLING FROM EACH MECHANISMS
########################################################################################################################################	

	def _laplace_mechanism(self, sensitivity):
		noised = [i + math.ceil(numpy.random.laplace(0, sensitivity/self._epsilon)) for i in self._observation_counts]
		noised = [self._sample_size if i > self._sample_size else i for i in noised]
		noised = [0 if i < 0 else i for i in noised]
		if self._sample_size - sum(noised[:-1]) < 0:
			noised[-1] = 0
		elif self._sample_size - sum(noised[:-1]) > self._sample_size:
			noised[-1] = self._sample_size
		else:
			noised[-1] = self._sample_size - sum(noised[:-1]) 
		self._laplaced_posterior = dirichlet(noised) + self._prior



	def _exponentialize(self):
		self._exponential_posterior = numpy.random.choice(self._candidates, p=self._GS_probabilities)


	def _exponentialize_LS(self):
		self._exponential_posterior = numpy.random.choice(self._candidates, p=self._LS_probabilities)

	def _exponentialize_SS(self):
		self._exponential_posterior = numpy.random.choice(self._candidates, p=self._SS_probabilities)

	def _exponentialize_alpha_SS(self):
		self._exponential_posterior = numpy.random.choice(self._candidates, p=self._alpha_SS_probabilities)


	def _propose_test_release(self):
		return

########################################################################################################################################
######EXPERMENTS FRO N TIMES, I.E., CALL THE SAMPLING FUNCTION OF EACH MECHANISMS FOR N TIMES
########################################################################################################################################	

	def _experiments(self, times):
		self._set_candidate_scores()
		self._set_local_sensitivities()

		self._set_up_baseline_lap_mech()
		self._set_up_improved_lap_mech()
		self._set_up_exp_mech_with_SS()
		self._set_up_exp_mech_with_alpha_SS()
		self._set_up_exp_mech_with_GS()
		self._set_up_exp_mech_with_LS()


		def sensitivity_for_lap(dimension):
			if (dimension == 2):
				return 1,2
			else:
				return 2,dimension
		s1,s2 = sensitivity_for_lap(self._prior._size)


		for i in range(times):
			#############################################################################
			#SAMPLING WITH THE BASELINE LAPLACE MECHANISM 
			#############################################################################
			self._laplace_mechanism(sensitivity = s2)
			self._accuracy[self._keys[0]].append(self._posterior - self._laplaced_posterior)
			
			#############################################################################
			#SAMPLING WITH THE IMRPOVED LAPLACE MECHANISM 
			#############################################################################
			self._laplace_mechanism(sensitivity = s1)
			self._accuracy[self._keys[1]].append(self._posterior - self._laplaced_posterior)

			#############################################################################
			#SAMPLING WITH THE EXPONENTIAL MECHANISM OF SMOOTH SENSITIVITY
			#############################################################################
			self._exponentialize_SS()
			self._accuracy[self._keys[2]].append(self._posterior - self._exponential_posterior)

			#############################################################################
			#SAMPLING WITH THE EXPONENTIAL MECHANISM OF ALPHA SMOOTH SENSITIVITY
			# #############################################################################
			self._exponentialize_alpha_SS()
			self._accuracy[self._keys[3]].append(self._posterior - self._exponential_posterior)

			#############################################################################
			#SAMPLING WITH THE EXPONENTIAL MECHANISM OF ALPHA SMOOTH SENSITIVITY
			# #############################################################################
			self._exponentialize()
			self._accuracy[self._keys[4]].append(self._posterior - self._exponential_posterior)
			
			#############################################################################
			#SAMPLING WITH THE EXPONENTIAL MECHANISM OF ALPHA SMOOTH SENSITIVITY
			# #############################################################################
			self._exponentialize_LS()
			self._accuracy[self._keys[5]].append(self._posterior - self._exponential_posterior)

			
		for key,item in self._accuracy.items():
			self._accuracy_mean[key] = numpy.mean(item)


########################################################################################################################################
######PRINT FUNCTION TO SHOW THE PRARMETERS OF THE CLASS
########################################################################################################################################	

	def _get_bias(self):
		return self._bias

	def _get_observation(self):
		return self._observation

	def _get_posterior(self):
		return self._posterior


	def _show_bias(self):
		print "The bias generated from the prior distribution is: " + str(self._bias)

	def _show_laplaced(self):
		print "The posterior distribution under Laplace mechanism is: "
		self._laplaced_posterior.show()


	def _show_observation(self):
		print "The observed data set is: "
		print self._observation
		print "The observed counting data is: "
		print self._observation_counts

	def _show_VS(self):
		print "The varying sensitivity for every candidate is:"
		for r in self._candidates:
			print r._alphas, self._VS[r]

	def _show_exponential(self):
		print "The posterior distribution under Exponential Mechanism is: "
		self._exponential_posterior.show()

	def _show_prior(self):
		print "The prior distribution is: "
		self._prior.show()

	def _show_all(self):
		self._show_prior()
		self._show_bias()
		self._show_observation()
		self._show_laplaced()
		self._show_exponential()
