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
	def __init__(self, prior, sample_size, epsilon, delta = 0.0000000001, gamma=1.0):
		self._prior = prior
		self._sample_size = sample_size
		self._epsilon = epsilon
		self._delta = delta
		self._gamma = gamma
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
		self._gamma_SS_probabilities = []
		self._LS_Candidates = {}
		self._GS = 0.0
		self._LS = 0.0
		self._SS = 0.0
		self._gamma_SS = 0.0
		self._keys = []
		self._accuracy = {}
		self._accuracy_mean = {}
	def _set_gamma(self, gamma):
		self._gamma = gamma
		
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
		self._set_candidates([], self._sample_size)

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

	def _set_up_naive_lap_mech(self):
		self._keys.append("Naive LaplaceMech")
		self._accuracy["Naive LaplaceMech"]=[]
		self._accuracy_mean["Naive LaplaceMech"]=[]

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
		eps = self._epsilon/(0.5 + 0.5 * self._gamma)
		for r in self._candidates:
			temp = math.exp(eps * self._candidate_scores[r]/(4.0 * self._SS))
			self._SS_probabilities.append(temp)
			nomalizer += temp

		for i in range(len(self._SS_probabilities)):
			self._SS_probabilities[i] = self._SS_probabilities[i]/nomalizer
		return nomalizer



	###################################################################################################################################
	#####SETTING UP THE EXPONENTIAL MECHANISM WITH THE ALPHA SMOOTH SENSITIVITY
	###################################################################################################################################	

	def _set_up_exp_mech_with_gamma_SS(self):

		###################################################################################################################################
		#INITIALIZE THE LISTS FOR EXPERIMENTS
		###################################################################################################################################		
		self._keys.append("ExpoMech of gamma SS")
		self._accuracy["ExpoMech of gamma SS"] = []
		self._accuracy_mean["ExpoMech of gamma SS"] = []

		###################################################################################################################################
		#CALCULATING THE SENSITIVITY
		###################################################################################################################################
		t0 = time.time()
		start = time.clock()
		self._gamma_SS = max([(1.0 / (1.0/self._LS_Candidates[r] + self._gamma * Hamming_Distance(self._observation_counts, [r._alphas[i] - self._prior._alphas[i] for i in range(self._prior._size)]))) for r in self._candidates])
		t1 = time.time()
		#print ("gamma smooth sensitivity"+str(t1 - t0)), self._gamma_SS


		###################################################################################################################################
		#CALCULATING THE SENSITIVITY
		###################################################################################################################################
		nomalizer = 0.0
		for r in self._candidates:
			temp = math.exp(self._epsilon * self._candidate_scores[r]/(4.0 * self._gamma_SS))
			self._gamma_SS_probabilities.append(temp)
			nomalizer += temp

		self._gamma_SS_probabilities = [i/nomalizer for i in self._gamma_SS_probabilities]
		
		return nomalizer



	###################################################################################################################################
	#####SETTING UP THE EXPONENTIAL MECHANISM WITH THE LOCAL SENSITIVITY
	###################################################################################################################################	
	def _set_up_exp_mech_with_LS(self):

		self._accuracy["Expomech of LS"] = []
		self._keys.append("Expomech of LS")

		
		self._LS = self._posterior._hellinger_sensitivity()

		self._LS = dirichlet([sum(self._posterior._alphas)/2, sum(self._posterior._alphas)/2])._hellinger_sensitivity()

		
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
		noised = [i + math.floor(numpy.random.laplace(0, sensitivity/self._epsilon)) for i in self._observation_counts]
		noised = [self._sample_size if i > self._sample_size else 0.0 if i < 0.0 else i for i in noised]
		r = self._sample_size - sum(noised[:-1])
		noised[-1] = 0.0 if r < 0.0 else self._sample_size if r > self._sample_size else r

		self._laplaced_posterior = dirichlet(noised) + self._prior


	def _laplace_mechanism_no_post(self, sensitivity):
		noised = []
		for a in self._posterior._alphas:
			t = a + numpy.random.laplace(0, sensitivity/self._epsilon)
			if t < 0.0:
				noised.append(0.0)
			else:
				noised.append(t)

		self._laplaced_posterior = dirichlet(noised)

	def _laplace_mechanism_naive(self):
		y1, y2 = numpy.random.laplace(0, 2.0/self._epsilon), numpy.random.laplace(0, 2.0/self._epsilon)
		noised =[0 if y1 < 0.0 else self._sample_size if y1 > self._sample_size else y1, 0 if y2 < 0.0 else self._sample_size if y2 > self._sample_size else y2]

		self._laplaced_posterior = dirichlet(noised) + self._prior
		
	def _laplace_mechanism_symetric(self, sensitivity):
		noised = [0.0,0.0]

		if(random.random() < 0.5):
			noised[0] = self._observation_counts[0] + math.floor(numpy.random.laplace(0, sensitivity/self._epsilon))
			noised[0] = 0.0 if noised[0] < 0.0 else noised[0]
			noised[0] = self._sample_size if noised[0] > self._sample_size else noised[0]
			noised[1] = self._sample_size - noised[0]
		else:
			noised[1] = self._observation_counts[1] + math.floor(numpy.random.laplace(0, sensitivity/self._epsilon))
			noised[1] = 0.0 if noised[1] < 0.0 else noised[1]
			noised[1] = self._sample_size if noised[1] > self._sample_size else noised[1]
			noised[0] = self._sample_size - noised[1]

		self._laplaced_posterior = dirichlet(noised) + self._prior


	def _exponentialize(self):
		self._exponential_posterior = numpy.random.choice(self._candidates, p=self._GS_probabilities)


	def _exponentialize_LS(self):
		self._exponential_posterior = numpy.random.choice(self._candidates, p=self._LS_probabilities)

	def _exponentialize_SS(self):
		self._exponential_posterior = numpy.random.choice(self._candidates, p=self._SS_probabilities)

	def _exponentialize_gamma_SS(self):
		self._exponential_posterior = numpy.random.choice(self._candidates, p=self._gamma_SS_probabilities)


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
		self._set_up_naive_lap_mech()
		# self._set_up_exp_mech_with_SS()
		self._set_up_exp_mech_with_gamma_SS()
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

			#SAMPLING WITH THE NAIVE LAPLACE MECHANISM 
			#############################################################################
			self._laplace_mechanism_naive()
			self._accuracy[self._keys[2]].append(self._posterior - self._laplaced_posterior)

			#############################################################################
			#SAMPLING WITH THE EXPONENTIAL MECHANISM OF SMOOTH SENSITIVITY
			#############################################################################
			# self._exponentialize_SS()
			# self._accuracy[self._keys[2]].append(self._posterior - self._exponential_posterior)

			#############################################################################
			#SAMPLING WITH THE EXPONENTIAL MECHANISM OF ALPHA SMOOTH SENSITIVITY
			# #############################################################################
			self._exponentialize_gamma_SS()
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
