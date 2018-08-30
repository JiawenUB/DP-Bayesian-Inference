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

def L1_Nrom(A, B):
	return numpy.sum(abs(numpy.array(A._alphas) - numpy.array(B._alphas)))

def multibeta_function(alphas):
	numerator = 1.0
	denominator = 0.0
	for alpha in alphas:
		numerator = numerator * math.gamma(alpha)
		denominator = denominator + alpha
	# print numerator / math.gamma(denominator)
	return numerator / math.gamma(denominator)

def optimized_multibeta_function(alphas):
	denominator = -1.0
	nominators = []
	denominators = []
	r = 1.0
	for alpha in alphas:
		denominator = denominator + alpha
	for alpha in alphas:
		# print alpha
		temp = alpha - 1
		while temp > 0.0:
			nominators.append(temp)
			temp -=1.0
		
		if temp < 0.0 and temp > -1.0:
			#print temp
			nominators.append(math.gamma(1 + temp))
	while denominator > 0.0:
		denominators.append(denominator)
		denominator -= 1.0
	if denominator < 0.0 and denominator > -1.0:
		denominators.append(math.gamma(1.0 + denominator))

	denominators.sort()
	nominators.sort()
	# print nominators
	# print denominators
	d_pointer = len(denominators) - 1
	n_pointer = len(nominators) - 1
	while d_pointer >= 0 and n_pointer >= 0:
		# print nominators[n_pointer],denominators[d_pointer]
		r *= nominators[n_pointer] / denominators[d_pointer]
		n_pointer -= 1
		d_pointer -= 1
	while d_pointer >= 0:
		# print n_pointer,denominators[d_pointer]
		r *= 1.0 / denominators[d_pointer]
		d_pointer -= 1
	while n_pointer >= 0:
		# print nominators[n_pointer] ,d_pointer
		r *= nominators[n_pointer] 
		n_pointer -= 1
	return r

def Hellinger_Distance_Dir(Dir1, Dir2):
	return math.sqrt(1 - multibeta_function((numpy.array(Dir1._alphas) + numpy.array(Dir2._alphas)) / 2.0)/ \
		math.sqrt(multibeta_function(Dir1._alphas) * multibeta_function(Dir2._alphas)))

def Optimized_Hellinger_Distance_Dir(Dir1, Dir2):
	# print optimized_multibeta_function((numpy.array(Dir1._alphas) + numpy.array(Dir2._alphas)) / 2.0)/ \
		# math.sqrt(optimized_multibeta_function(Dir1._alphas) * optimized_multibeta_function(Dir2._alphas))
	# print Dir1._alphas,Dir2._alphas
	return math.sqrt(1 - optimized_multibeta_function((numpy.array(Dir1._alphas) + numpy.array(Dir2._alphas)) / 2.0)/ \
		math.sqrt(optimized_multibeta_function(Dir1._alphas) * optimized_multibeta_function(Dir2._alphas)))

def Hamming_Distance(Dir1, Dir2):
	temp = [abs(a - b) for a,b in zip(Dir1._alphas,Dir2._alphas)]
	return sum(temp)

class Dir(object):
	def __init__(self, alphas):
		self._alphas = alphas
		self._size = len(alphas)

	def __sub__(self, other):
		return Optimized_Hellinger_Distance_Dir(self, other)

	def __add__(self, other):
		return Dir(list(numpy.array(self._alphas) + numpy.array(other._alphas)))

	def show(self):
		print "Dirichlet("+str(self._alphas) + ")"

	def _hellinger_sensitivity(self,r):
		LS = 0.0
		temp = deepcopy(r._alphas)
		for i in range(0, self._size-1):
			temp[i] += 1
			# print temp
			for j in range(i + 1, self._size):
				temp[j] -= 1
				# print temp
				if temp[j]<=0:
					temp[j] += 1
					continue
				LS = max(LS, abs(Dir(temp) - self))
				# print r._alphas,self._alphas,temp,(r-self),(Dir(temp) - self)
				temp[j] += 1
			temp[i] -= 1
		return LS


	def _score_sensitivity(self, r):
		LS = 0.0
		temp = deepcopy(self._alphas)
		for i in range(0, self._size):
			temp[i] += 1
			for j in range(i + 1, self._size):
				temp[j] -= 1
				LS = max(LS, abs(-(r - self) + (r - Dir(temp))))
				temp[j] += 1
			temp[i] -= 1
		return LS


class BayesInferwithDirPrior(object):
	def __init__(self, prior, sample_size, epsilon, delta):
		self._prior = prior
		self._sample_size = sample_size
		self._epsilon = epsilon
		self._delta = delta
		self._bias = numpy.random.dirichlet(self._prior._alphas)
		self._observation = numpy.random.multinomial(1, self._bias, self._sample_size)
		self._observation_counts = numpy.sum(self._observation, 0)
		self._posterior = Dir(self._observation_counts) + self._prior
		self._laplaced_posterior = self._posterior
		self._exponential_posterior = self._posterior
		self._SS_posterior = self._posterior
		self._candidate_scores = {}
		self._candidates = []
		self._LS_probabilities = []
		self._GS_probabilities = []
		self._SS_probabilities = []
		self._GS = 0.0
		self._LS_Candidates = {}
		self._VS = {}
		self._SS = {}
		self._LS = 0.0
		self._SS_Hamming = 0.0
		# self._SS_Expon = 0.0
		# self._SS_Laplace = 0.0
		self._candidate_VS_scores = {}
		#self._keys = ["Laplace Mechanism | Achieving" + str(self._epsilon) + "-DP"]
		self._keys = ["LaplaceMech"]
		self._accuracy = {self._keys[0]:[]}
		self._accuracy_l1 = {self._keys[0]:[]}
	
	def _set_bias(self, bias):
		self._bias = bias
		self._update_observation()

	def _set_observation(self,observation):
		self._posterior = Dir(observation) + self._prior

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
		self._posterior = Dir(self._observation_counts) + self._prior

	def _set_candidate_scores(self):
		# print "Calculating Candidates and Scores....."
		start = time.clock()
		self._set_candidates([], numpy.sum(self._observation))

		for r in self._candidates:
			self._candidate_scores[r] = -(self._posterior - r)
		# print str(time.clock() - start) + " seconds."

	def _set_candidates(self, current, rest):
		if len(current) == len(self._prior._alphas) - 1:
			current.append(rest)
			self._candidates.append(Dir(deepcopy(current)) + self._prior)
			current.pop()
			return
		for i in range(0, rest + 1):
			current.append(i)
			self._set_candidates(current, rest - i)
			current.pop()

	def _set_LS_Candidates(self):
		for r in self._candidates:
			self._LS_Candidates[r] = r._hellinger_sensitivity(r)

	def _set_SS(self):

		self._set_LS_Candidates()
		start = time.clock()
		beta = math.log(1 - self._epsilon / (2.0 * math.log(self._delta / (2.0 * (self._sample_size)))))
		self._SS_Hamming = max(self._LS, max([self._LS_Candidates[r] * math.exp(- beta * Hamming_Distance(self._posterior, r)) for r in self._candidates]))

		key3 = "Exponential Mechanism with " + str(beta) + " - Bound Smooth Sensitivity (" + str(self._SS_Hamming) + ")|(" + str(self._epsilon) + "," + str(self._delta) + ")-DP"
		# print key3
		key3 = "ExpoMech of SS"
		self._accuracy[key3] = []
		self._accuracy_l1[key3] = []
		self._keys.append(key3)
		# print str(time.clock() - start) + "seconds."

		nomalizer = 0.0


		for r in self._candidates:
			temp = math.exp(self._epsilon * self._candidate_scores[r]/(self._SS_Hamming))
			self._SS_probabilities.append(temp)

			nomalizer += temp

		for i in range(len(self._SS_probabilities)):
			self._SS_probabilities[i] = self._SS_probabilities[i]/nomalizer

		return nomalizer


	def _set_LS(self):
		self._LS = self._posterior._hellinger_sensitivity(self._posterior)#self._posterior
		key = "Exponential Mechanism with Local Sensitivity - " + str(self._LS) + "| Non Privacy"
		# print key
		key = "Expomech of LS"
		self._accuracy[key] = []
		self._accuracy_l1[key] = []
		self._keys.append(key)

		nomalizer = 0.0

		for r in self._candidates:
			temp = math.exp(self._epsilon * self._candidate_scores[r]/(self._LS))
			self._LS_probabilities.append(temp)
			nomalizer += temp

		for i in range(len(self._LS_probabilities)):
			self._LS_probabilities[i] = self._LS_probabilities[i]/nomalizer

		return nomalizer


	def _set_GS(self):
		t1 = [1 for i in range(self._prior._size)]
		t1[0] += 1
		t2 = [1 for i in range(self._prior._size)]
		t2[1] += 1
		
		self._GS = Dir(t1) - Dir(t2)
		key = "Exponential Mechanism with Global Sensitivity - " + str(self._GS) + "| Achieving" + str(self._epsilon) + "-DP"
		# print key
		key = "ExpoMech of GS"
		self._accuracy[key] = []
		self._accuracy_l1[key] = []
		self._keys.append(key)

		nomalizer = 0.0
		
		for r in self._candidates:
			temp = math.exp(self._epsilon * self._candidate_scores[r]/(self._GS))
			self._GS_probabilities.append(temp)
			nomalizer += temp

		for i in range(len(self._GS_probabilities)):
			self._GS_probabilities[i] = self._GS_probabilities[i]/nomalizer

		return nomalizer	



	def _laplace_noize(self):
		self._laplaced_posterior = Dir([alpha + math.floor(numpy.random.laplace(0, 2.0/self._epsilon)) for alpha in self._posterior._alphas])

	def _laplace_noize_navie(self):
		t = numpy.random.laplace(0, 1.0/self._epsilon)
		self._laplaced_posterior = Dir([self._posterior._alphas[0] + t, self._posterior._alphas[0] - t])


	def _laplace_noize_mle(self):
		while True:
			flage = True
			self._laplaced_posterior = Dir([alpha + round(numpy.random.laplace(0, 2.0/self._epsilon)) for alpha in self._posterior._alphas])
			self._laplaced_posterior._alphas[0] += (sum(self._prior._alphas) + self._sample_size - sum(self._laplaced_posterior._alphas))
			for  alpha in self._laplaced_posterior._alphas:
				if alpha <= 0.0:
					flage = False
			if flage:
				break

	def _exponentialize_GS(self):
		probabilities = {}
		self._exponential_posterior = numpy.random.choice(self._candidates, p=self._GS_probabilities)


	def _exponentialize_LS(self):
		probabilities = {}
		self._exponential_posterior = numpy.random.choice(self._candidates, p=self._LS_probabilities)


	def _exponentialize_SS(self):
		probabilities = {}
		nomalizer = 0.0
		self._exponential_posterior = numpy.random.choice(self._candidates, p=self._SS_probabilities)


	def _experiments(self, times):
		self._set_candidate_scores()
		self._set_GS()
		self._set_LS()
		self._set_SS()

		for i in range(times):
			self._laplace_noize()
			self._accuracy[self._keys[0]].append(self._posterior - self._laplaced_posterior)

			self._exponentialize_GS()
			self._accuracy[self._keys[1]].append(self._posterior - self._exponential_posterior)

			self._exponentialize_LS()
			self._accuracy[self._keys[2]].append(self._posterior - self._exponential_posterior)

			self._exponentialize_SS()
			self._accuracy[self._keys[3]].append(self._posterior - self._exponential_posterior)




def accuracy_study_discrete(sample_size,epsilon,delta,prior,observation):
	Bayesian_Model = BayesInferwithDirPrior(prior, sample_size, epsilon, delta)
	Bayesian_Model._set_observation(observation)

	Bayesian_Model._set_candidate_scores()
	Bayesian_Model._set_LS()
	nomalizer = Bayesian_Model._set_SS()

	sorted_scores = sorted(Bayesian_Model._candidate_scores.items(), key=operator.itemgetter(1))
	steps = [-i for i in sorted(list(set(Bayesian_Model._candidate_scores.values())))]

	Bayesian_Model._SS_probabilities.sort()
	# print Bayesian_Model._SS_probabilities

	i = 0
	# print sorted_scores

	candidates_classfied_by_steps = []

	probabilities_exp_by_steps = []
	probabilities_lap_by_steps = []

	print "Here are probabilities w.r.t. steps (Hellinegr) of Exponential Mechanism with Smooth sensitivity"

	while i < len(sorted_scores):
		j = i
		candidates_for_print = []
		candidates_for_classify = []
		while True:
			if (i+1) > len(sorted_scores) or sorted_scores[j][1] != sorted_scores[i][1]:
				break
			candidates_for_print.append(sorted_scores[i][0]._alphas)
			candidates_for_classify.append(sorted_scores[i][0])
			i += 1
		candidates_classfied_by_steps.append(candidates_for_classify)
		probabilities_exp_by_steps.append(Bayesian_Model._SS_probabilities[j]*(i - j))
		print "Pr[H(BI(x), r) = " + str(-sorted_scores[j][1]) + " ] = " + str(Bayesian_Model._SS_probabilities[j]*(i - j)) + " (r = " + str(candidates_for_print) +")"
   



data_size = 9
epsilon = 0.8
delta = 0.0005

# Dir is the class name of dirichlet distribution
# we use it to create an object of prior distribution, when dimension is two, it is a beta distribution
# The initial parameter is the parameter of the prior distribution

prior = Dir([1,1,1])

observation = [3,3,3]

accuracy_study_discrete(data_size,epsilon,delta,prior,observation)
	
