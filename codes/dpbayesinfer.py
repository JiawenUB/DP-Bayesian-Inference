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
		self._GS = 0.0
		self._LS_Candidates = {}
		self._VS = {}
		self._SS = {}
		self._LS = 0.0
		self._SS = 0.0
		# self._SS_Expon = 0.0
		# self._SS_Laplace = 0.0
		self._candidate_VS_scores = {}
		#self._keys = ["Laplace Mechanism | Achieving" + str(self._epsilon) + "-DP"]
		self._keys = ["LaplaceMech"]
		self._accuracy = {self._keys[0]:[]}
		self._accuracy_l1 = {self._keys[0]:[]}
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

	def _set_LS_Candidates(self):
		for r in self._candidates:
			self._LS_Candidates[r] = r._hellinger_sensitivity()

	def _set_SS_opt(self):
		self._set_LS_Candidates()
		start = time.clock()
		beta = math.log(1 - self._epsilon / (2.0 * math.log(self._delta / (2.0 * (self._sample_size)))))
		for i in range(n):
			extrem_values = []
		self._SS = max(self._LS, max([self._LS_Candidates[r] * math.exp(- beta * Hamming_Distance(self._observation_counts, [r._alphas[i] - self._prior._alphas[i] for i in range(self._prior._size)])) for r in self._candidates]))
		key3 = "Exponential Mechanism with " + str(beta) + " - Bound Smooth Sensitivity (" + str(self._SS) + ")|(" + str(self._epsilon) + "," + str(self._delta) + ")-DP"
		key3 = "ExpoMech of SS"
		self._accuracy[key3] = []
		self._accuracy_l1[key3] = []
		self._keys.append(key3)
		nomalizer = 0.0


		for r in self._candidates:
			temp = math.exp(self._epsilon * self._candidate_scores[r]/(2 * self._SS))
			self._SS_probabilities.append(temp)
			nomalizer += temp

		for i in range(len(self._SS_probabilities)):
			self._SS_probabilities[i] = self._SS_probabilities[i]/nomalizer
			# print self._candidates[i]._alphas, self._SS_probabilities[i]

		return nomalizer



	def _set_SS(self):

		t0 = time.time()
		self._set_LS_Candidates()
		start = time.clock()
		beta = math.log(1 - self._epsilon / (2.0 * math.log(self._delta / (2.0 * (self._sample_size)))))
		self._SS = max(self._LS, max([self._LS_Candidates[r] * math.exp(- beta * Hamming_Distance(self._observation_counts, [r._alphas[i] - self._prior._alphas[i] for i in range(self._prior._size)])) for r in self._candidates]))
		key3 = "Exponential Mechanism with " + str(beta) + " - Bound Smooth Sensitivity (" + str(self._SS) + ")|(" + str(self._epsilon) + "," + str(self._delta) + ")-DP"
		key3 = "ExpoMech of SS"
		self._accuracy[key3] = []
		self._accuracy_l1[key3] = []
		self._keys.append(key3)
		t1 = time.time()
		print("smooth sensitivity"+str(t1 - t0))

		# beta = 0.0
		# self._SS_Expon = max(self._LS, max([self._LS_Candidates[r] * math.exp(- beta * Hamming_Distance(self._posterior, r)) for r in self._candidates]))
		# key2 = "Exponential Mechanism with " + str(beta) + " - Bound Smooth Sensitivity - " + str(self._SS_Expon) + "| Achieving" + str(self._epsilon) + "-DP"
		# self._accuracy[key2] = []
		# self._keys.append(key2)

	def _set_SS_probabilities(self):
		nomalizer = 0.0
		for r in self._candidates:
			temp = math.exp(self._epsilon * self._candidate_scores[r]/(2 * self._SS))
			self._SS_probabilities.append(temp)
			nomalizer += temp

		for i in range(len(self._SS_probabilities)):
			self._SS_probabilities[i] = self._SS_probabilities[i]/nomalizer
			# print self._candidates[i]._alphas, self._SS_probabilities[i]
		return nomalizer


	def _set_LS(self):
		t0 = time.time()
		self._LS = self._posterior._hellinger_sensitivity()#self._posterior

		key = "Exponential Mechanism with Local Sensitivity - " + str(self._LS) + "| Non Privacy"
		# print key
		key = "Expomech of LS"
		self._accuracy[key] = []
		self._accuracy_l1[key] = []
		self._keys.append(key)

		nomalizer = 0.0
		# print self._posterior._alphas

		for r in self._candidates:
			temp = math.exp(self._epsilon * self._candidate_scores[r]/(2 * self._LS))
			self._LS_probabilities.append(temp)
			nomalizer += temp

		for i in range(len(self._LS_probabilities)):
			self._LS_probabilities[i] = self._LS_probabilities[i]/nomalizer
		t1 = time.time()
		print("local sensitivity"+str(t1 - t0))
		return nomalizer




	def _set_GS(self):
		t0 = time.time()
		t1 = [1 for i in range(self._prior._size)]
		t1[0] += 1
		t2 = [1 for i in range(self._prior._size)]
		t2[1] += 1
		
		self._GS = dirichlet(t1) - dirichlet(t2)
		key = "Exponential Mechanism with Global Sensitivity - " + str(self._GS) + "| Achieving" + str(self._epsilon) + "-DP"
		# print key
		key = "ExpoMech of GS"
		self._accuracy[key] = []
		self._accuracy_l1[key] = []
		self._keys.append(key)

		nomalizer = 0.0
		
		for r in self._candidates:
			temp = math.exp(self._epsilon * self._candidate_scores[r]/(2 * self._GS))
			self._GS_probabilities.append(temp)
			nomalizer += temp

		for i in range(len(self._GS_probabilities)):
			self._GS_probabilities[i] = self._GS_probabilities[i]/nomalizer
		t1 = time.time()
		print("global sensitivity"+str(t1 - t0))
		return nomalizer




	# def _set_VS(self):
	# 	t = 2 * math.log(len(self._candidates) / 0.8) / self._epsilon
	# 	print "Calculating Varying Sensitivity Scores....."
	# 	start = time.clock()
	# 	self._set_LS()
	# 	for r in self._candidates:
	# 		self._candidate_VS_scores[r] = -max([((-self._candidate_scores[r] + t * self._LS_Candidates[r] - (-self._candidate_scores[i] + t * self._LS_Candidates[i]))/(self._LS_Candidates[r] + self._LS_Candidates[i])) for i in self._candidates])
	# 	key = "Exponential Mechanism with Varying Sensitivity Scores | Achieving " + str(self._epsilon) + "-DP"
	# 	print key
	# 	key = "ExpoMech of VS"
	# 	self._accuracy[key] = []
	# 	self._accuracy_l1[key] = []
	# 	self._keys.append(key)
	# 	print str(time.clock() - start) + "seconds."

	# def _Smooth_Sensitivity_Noize(self):
	# 	gamma = 1
	# 	z = numpy.random.standard_cauchy()
	# 	alpha = self._epsilon/ (2.0 * (gamma + 1))
	# 	temp = [a + self._SS * z /alpha for a in self._posterior._alphas]
	# 	self._SS_posterior = dirichlet(temp)
	# 	return

	# def _Smooth_Sensitivity_Noize_Hamming(self):
	# 	gamma = 1
	# 	z = abs(numpy.random.standard_cauchy())
	# 	alpha = self._epsilon/ (2.0 * (gamma + 1))
	# 	temp = [a + self._SS * z /alpha for a in self._posterior._alphas]
	# 	self._SS_posterior = dirichlet(temp)
	# 	return

	# def _Smooth_Sensitivity_Laplace_Noize(self):
	# 	gamma = 1
	# 	Z = [numpy.random.laplace(0,1) for i in range(len(self._posterior._alphas))]
	# 	alpha = self._epsilon/ 2.0
	# 	temp = [a + self._SS_Laplace * z /alpha for a,z in self._posterior._alphas,Z]
	# 	self._SS_posterior = dirichlet(temp)
	# 	return		

	def _laplace_noize(self, sensitivity = 2.0):
		noised_alphas = []
		rest = self._sample_size
		for i in range(self._prior._size - 1):
			t = self._observation_counts[i] + math.floor(numpy.random.laplace(0, sensitivity/self._epsilon))
			if t < 0:
				noised_alphas.append(0)
				rest -= 0
			elif t > rest:
				noised_alphas.append(rest)
				rest -= rest
				for j in range(i+1,self._prior._size - 1):
					noised_alphas.append(0)
				break
			else:
				noised_alphas.append(t)
				rest -= t
		noised_alphas.append(rest)
		self._laplaced_posterior = dirichlet(noised_alphas) + self._prior


	# def _laplace_noize_ours(self):

	# 	self._laplaced_posterior = dirichlet([alpha + math.floor(numpy.random.laplace(0, 1.0/self._epsilon)) for alpha in self._posterior._alphas])

	def _laplace_fourier(self):
		
		
		return

	def _laplace_noize_navie(self):
		t = numpy.random.laplace(0, 1.0/self._epsilon)
		self._laplaced_posterior = dirichlet([self._posterior._alphas[0] + t, self._posterior._alphas[0] - t])

	def _laplace_noize_zhang(self):
		noised = [i + numpy.random.laplace(0, 2.0/self._epsilon) for i in self._observation_counts]
		noised = [self._sample_size if i > self._sample_size else i for i in noised]
		noised = [0 if i < 0 else i for i in noised]
		self._laplaced_zhang_posterior = dirichlet(noised) + self._prior



	# def _laplace_noize_mle(self):
	# 	while True:
	# 		flage = True
	# 		self._laplaced_posterior = dirichlet([alpha + round(numpy.random.laplace(0, 2.0/self._epsilon)) for alpha in self._posterior._alphas])
	# 		self._laplaced_posterior._alphas[0] += (sum(self._prior._alphas) + self._sample_size - sum(self._laplaced_posterior._alphas))
	# 		for  alpha in self._laplaced_posterior._alphas:
	# 			if alpha <= 0.0:
	# 				flage = False
	# 		if flage:
	# 			break

	def _exponentialize_GS(self):
		probabilities = {}
		self._exponential_posterior = numpy.random.choice(self._candidates, p=self._GS_probabilities)


	def _exponentialize_LS(self):
		probabilities = {}
		self._exponential_posterior = numpy.random.choice(self._candidates, p=self._LS_probabilities)


 		self._exponential_posterior = r

	def _exponentialize_SS(self):
		nomalizer = 0.0
		self._exponential_posterior = numpy.random.choice(self._candidates, p=self._SS_probabilities)

		# for r in self._candidates:
		# 	probabilities[r] = math.exp(self._epsilon * self._candidate_scores[r]/(self._SS))
		# 	nomalizer += probabilities[r]
		# outpro = random.random()
		# for r in self._candidates:
		# 	if outpro < 0:
		# 		return
		# 	outpro = outpro - probabilities[r]/nomalizer
		# 	self._exponential_posterior = r

	def _propose_test_release(self):
		return


	def _experiments(self, times):
		self._set_candidate_scores()
		self._set_GS()
		self._set_LS()
		self._set_SS()
		self._set_SS_probabilities()
		self._keys.append('Laplace_s2')
		self._accuracy['Laplace_s2'] = []
		self._accuracy_mean['Laplace_s2'] = []
		#self._show_all()
		for i in range(times):
			self._laplace_noize(sensitivity = 1.0)
			# print self._laplaced_posterior._alphas
			self._accuracy[self._keys[0]].append(self._posterior - self._laplaced_posterior)
			# self._laplace_noize()
			# self._accuracy_l1[self._keys[0]].append(L1_Nrom(self._posterior, self._laplaced_posterior))
			# self._exponentialize_GS()
			# self._accuracy_l1[self._keys[1]].append(L1_Nrom(self._posterior, self._exponential_posterior))
			# self._accuracy[self._keys[1]].append(self._posterior - self._exponential_posterior)
			# self._exponentialize_LS()
			# self._accuracy[self._keys[2]].append(self._posterior - self._exponential_posterior)
			# self._accuracy_l1[self._keys[2]].append(L1_Nrom(self._posterior, self._exponential_posterior))
			# self._Smooth_Sensitivity_Noize()
			# self._accuracy[self._keys[2]].append(self._posterior - self._SS_posterior)
			# self._Smooth_Sensitivity_Noize_Hamming()
			# self._accuracy[self._keys[3]].append(self._posterior - self._SS_posterior)
			# self._Smooth_Sensitivity_Laplace_Noize()
			# self._accuracy[self._keys[4]].append(self._posterior - self._SS_posterior)
			self._exponentialize_SS()
			self._accuracy[self._keys[3]].append(self._posterior - self._exponential_posterior)
			self._laplace_noize(sensitivity = 2.0)
			# print self._laplaced_posterior._alphas
			self._accuracy[self._keys[4]].append(self._posterior - self._laplaced_posterior)
			
			# self._laplace_noize_zhang()
			# self._accuracy[self._keys[4]].append(self._posterior - self._laplaced_zhang_posterior)
			# # self._accuracy_l1[self._keys[3]].append(L1_Nrom(self._posterior, self._exponential_posterior))
		for key,item in self._accuracy.items():
			self._accuracy_mean[key] = sum(item)*1.0/(1 if len(item) == 0 else len(item))



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
