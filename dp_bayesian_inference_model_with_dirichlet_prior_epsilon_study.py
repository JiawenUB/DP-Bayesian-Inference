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
from scipy.special import gammaln

def gen_betaln (alphas):
        numerator=0.0
        for alpha in alphas:
	        numerator = numerator + gammaln(alpha)
                
        return(numerator/math.log(float(sum(alphas))))


def opt_hellinger2(alphas, betas):
        z=gen_betaln(numpy.divide(numpy.sum([alphas, betas], axis=0), 2.0))-0.5*(gen_betaln(alphas) + gen_betaln(betas))
        return (math.sqrt(1-math.exp(z)))



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
	# 	math.sqrt(optimized_multibeta_function(Dir1._alphas) * optimized_multibeta_function(Dir2._alphas))
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
		return opt_hellinger2(self._alphas, other._alphas)
		# return Optimized_Hellinger_Distance_Dir(self, other)

	def _minus(self,other):
		self._alphas = list(numpy.array(self._alphas) - numpy.array(other._alphas))

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
		for i in range(0, self._size-1):
			temp[i] -= 1
			if temp[i]<=0:
					temp[i] += 1
					continue
			# print temp
			for j in range(i + 1, self._size):
				temp[j] += 1
				# print temp
				LS = max(LS, abs(Dir(temp) - self))
				# print r._alphas,self._alphas,temp,(r-self),(Dir(temp) - self)
				temp[j] -= 1
			temp[i] += 1		
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
		self._SS = 0.0
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
		self._observation_counts = observation
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
		# print "Calculating Smooth Sensitivity with Hellinger Distance....."
		# start = time.clock()
		#gamma = 1
		self._set_LS_Candidates()
		# beta = self._epsilon / (2.0 * len(self._prior._alphas) * (gamma + 1))
		# self._SS = max([self._LS_Candidates[r] * math.exp(- beta * Optimized_Hellinger_Distance_Dir(self._posterior, r)) for r in self._candidates])
		# key1 = "(" + str(self._epsilon / 2.0) + "," + str(beta) + ") Admissible Niose and " + str(beta) + "-Smooth Sensitivity (" + str(self._SS) + ")|" + str(self._epsilon) + "-DP"
		# self._accuracy[key1] = []
		# self._keys.append(key1)
		# print str(time.clock() - start) + "seconds."
		# print "Calculating Smooth Sensitivity with Hamming Distance....."
		# start = time.clock()
		# self._SS = max([self._LS_Candidates[r] * math.exp(- beta * Hamming_Distance(self._posterior, r)) for r in self._candidates])
		# key2 = "(" + str(self._epsilon / 2.0) + "," + str(beta) + ") Admissible Niose and " + str(beta) + "-Smooth Sensitivity (" + str(self._SS) + ")|" + str(self._epsilon) + "-DP"
		# self._accuracy[key2] = []
		# self._keys.append(key2)
		# print str(time.clock() - start) + "seconds."
		# print "Calculating Smooth Sensitivity with Hamming Distance....."
		start = time.clock()
		beta = math.log(1 - self._epsilon / (2.0 * math.log(self._delta / (2.0 * (self._sample_size)))))
		self._SS = max(self._LS, max([self._LS_Candidates[r] * math.exp(- beta * Hamming_Distance(self._posterior, r)) for r in self._candidates]))
		# self._SS = self._LS
		key3 = "Exponential Mechanism with " + str(beta) + " - Bound Smooth Sensitivity (" + str(self._SS) + ")|(" + str(self._epsilon) + "," + str(self._delta) + ")-DP"
		# print key3
		key3 = "ExpoMech of SS"
		self._accuracy[key3] = []
		self._accuracy_l1[key3] = []
		self._keys.append(key3)
		# print str(time.clock() - start) + "seconds."

		nomalizer = 0.0


		for r in self._candidates:
			temp = math.exp(self._epsilon * self._candidate_scores[r]/(2 * self._SS))
			self._SS_probabilities.append(temp)

			nomalizer += temp

		for i in range(len(self._SS_probabilities)):
			self._SS_probabilities[i] = self._SS_probabilities[i]/nomalizer
			# print self._candidates[i]._alphas, self._SS_probabilities[i]

		return nomalizer
		# beta = 0.0
		# self._SS_Expon = max(self._LS, max([self._LS_Candidates[r] * math.exp(- beta * Hamming_Distance(self._posterior, r)) for r in self._candidates]))
		# key2 = "Exponential Mechanism with " + str(beta) + " - Bound Smooth Sensitivity - " + str(self._SS_Expon) + "| Achieving" + str(self._epsilon) + "-DP"
		# self._accuracy[key2] = []
		# self._keys.append(key2)

	def _set_LS(self):
		self._LS = self._posterior._hellinger_sensitivity(self._posterior)#self._posterior

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
			temp = math.exp(self._epsilon * self._candidate_scores[r]/(2 * self._GS))
			self._GS_probabilities.append(temp)
			nomalizer += temp

		for i in range(len(self._GS_probabilities)):
			self._GS_probabilities[i] = self._GS_probabilities[i]/nomalizer

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
	# 	self._SS_posterior = Dir(temp)
	# 	return

	# def _Smooth_Sensitivity_Noize_Hamming(self):
	# 	gamma = 1
	# 	z = abs(numpy.random.standard_cauchy())
	# 	alpha = self._epsilon/ (2.0 * (gamma + 1))
	# 	temp = [a + self._SS * z /alpha for a in self._posterior._alphas]
	# 	self._SS_posterior = Dir(temp)
	# 	return

	# def _Smooth_Sensitivity_Laplace_Noize(self):
	# 	gamma = 1
	# 	Z = [numpy.random.laplace(0,1) for i in range(len(self._posterior._alphas))]
	# 	alpha = self._epsilon/ 2.0
	# 	temp = [a + self._SS_Laplace * z /alpha for a,z in self._posterior._alphas,Z]
	# 	self._SS_posterior = Dir(temp)
	# 	return		

	def _laplace_noize(self):
		noised_alphas = []
		rest = self._sample_size
		for i in range(self._prior._size - 1):
			alpha = self._posterior._alphas[i]
			temp = math.floor(numpy.random.laplace(0, 2.0/self._epsilon))
			if temp < - self._observation_counts[i]:
				noised_alphas.append(0)
				rest -= 0
			elif (self._observation_counts[i] + temp) > rest:
				noised_alphas.append(rest)
				rest -= rest
				for j in range(i+1,self._prior._size - 1):
					noised_alphas.append(0)
				break
			else:
				noised_alphas.append((self._observation_counts[i] + temp))
				rest -= (self._observation_counts[i] + temp)
		noised_alphas.append(rest)
		# print noised_alphas
		self._laplaced_posterior = Dir(noised_alphas) + self._prior


	def _laplace_noize_n(self):

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


	# def _exponentialize_VS(self):
	# 	probabilities = {}
	# 	for r in self._candidates:
	# 		probabilities[r] = math.exp(self._epsilon * self._candidate_VS_scores[r]/(1.0))
	# 		nomalizer += probabilities[r]
	# 	for r in self._candidates:
	# 		probabilities[r] = probabilities[r]	/ nomalizer	

	# 	self._exponential_posterior = numpy.random.choice(self._candidates, p=probabilities.valuse())
	# 	outpro = random.random()
	# 	for r in self._candidates:
	# 		if outpro < 0:
	# 			return
	# 		outpro = outpro - probabilities[r]/nomalizer
	# 		self._exponential_posterior = r

	def _exponentialize_SS(self):
		probabilities = {}
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
		#self._show_all()
		for i in range(times):
			self._laplace_noize()
			# print self._laplaced_posterior._alphas
			self._accuracy[self._keys[0]].append(self._posterior - self._laplaced_posterior)
			# self._laplace_noize()
			# self._accuracy_l1[self._keys[0]].append(L1_Nrom(self._posterior, self._laplaced_posterior))
			self._exponentialize_GS()
			# self._accuracy_l1[self._keys[1]].append(L1_Nrom(self._posterior, self._exponential_posterior))
			self._accuracy[self._keys[1]].append(self._posterior - self._exponential_posterior)
			self._exponentialize_LS()
			self._accuracy[self._keys[2]].append(self._posterior - self._exponential_posterior)
			# self._accuracy_l1[self._keys[2]].append(L1_Nrom(self._posterior, self._exponential_posterior))
			# self._Smooth_Sensitivity_Noize()
			# self._accuracy[self._keys[2]].append(self._posterior - self._SS_posterior)
			# self._Smooth_Sensitivity_Noize_Hamming()
			# self._accuracy[self._keys[3]].append(self._posterior - self._SS_posterior)
			# self._Smooth_Sensitivity_Laplace_Noize()
			# self._accuracy[self._keys[4]].append(self._posterior - self._SS_posterior)
			self._exponentialize_SS()
			self._accuracy[self._keys[3]].append(self._posterior - self._exponential_posterior)
			# self._accuracy_l1[self._keys[3]].append(L1_Nrom(self._posterior, self._exponential_posterior))


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


def draw_error_1(errors, model):
	plt.subplots(nrows=len(errors), ncols=1, figsize=(18, len(errors) * 5.0))
	plt.tight_layout(pad=2, h_pad=4, w_pad=2, rect=None)
	rows = 1
	for key,item in errors.items():
		plt.subplot(len(errors), 1, rows)
		x = numpy.arange(0, len(item), 1)
		plt.axhline(y=numpy.mean(item), color='r', linestyle = '--', alpha = 0.8, label = "average error",linewidth=3)
		plt.scatter(x, numpy.array(item), s = 40, c = 'b', marker = 'o', alpha = 0.7, edgecolors='white', label = " error")
		plt.ylabel('Hellinger Distance')
		plt.xlabel('Runs (Bias = ' + str(model._bias) + ')')
		plt.title(key + ' (Data Size = ' + str(model._sample_size) + ', Global epsilon = ' + str(model._epsilon) + ')')
		plt.legend(loc="best")
		rows = rows + 1
		plt.ylim(-0.1,1.0)
		plt.xlim(0.0,len(item)*1.0)
		plt.grid()
	plt.show()
	plt.savefig("beta-GS-SS-LS-size300-runs200-epsilon0-5.png")
	return 

def draw_error(errors, model, filename):
	# plt.subplots(nrows=len(errors), ncols=1, figsize=(18, len(errors) * 5.0))
	# plt.tight_layout(pad=2, h_pad=4, w_pad=2, rect=None)
	rows = 1
	data = []
	title = []
	for key,item in errors.items():
		data.append(item)
		title.append(key)
		# plt.subplot(len(errors), 1, rows)
		# x = numpy.arange(0, len(item), 1)
		# plt.axhline(y=numpy.mean(item), color='r', linestyle = '--', alpha = 0.8, label = "average error",linewidth=3)
		# plt.scatter(x, numpy.array(item), s = 40, c = 'b', marker = 'o', alpha = 0.7, edgecolors='white', label = " error")
		# plt.ylabel('Hellinger Distance')
		# plt.xlabel('Runs (Bias = ' + str(model._bias) + ')')
		# plt.title(key + ' (Data Size = ' + str(model._sample_size) + ', Global epsilon = ' + str(model._epsilon) + ')')
		# plt.legend(loc="best")
		# rows = rows + 1
		# plt.ylim(-0.1,1.0)
		# plt.xlim(0.0,len(item)*1.0)
		# plt.grid()
	fig, ax = plt.subplots()
	bplot = plt.boxplot(data, notch=1, widths=0.4, sym='+', vert=2, whis=1.5,patch_artist=True)
	plt.xlabel("different mechanisms")
	plt.ylabel('Distance Based on Hellinger Distance')
	#ax.set_xlim(0.5, len(errors) + 0.5)

	plt.xticks(range(1, len(errors)+1),title)
	plt.title('Accuracy / data set:' + str(model._observation_counts) + ", posterior: " + str(model._posterior._alphas) + ", epsilon:" + str(model._epsilon))
	for box in bplot["boxes"]:
		box.set(color='navy', linewidth=1.5)
		box.set(facecolor='lightblue')
	plt.show()
	# plt.savefig(filename)
	return 

def draw_error_l1(errors, model, filename):
	# plt.subplots(nrows=len(errors), ncols=1, figsize=(18, len(errors) * 5.0))
	# plt.tight_layout(pad=2, h_pad=4, w_pad=2, rect=None)
	rows = 1
	data = []
	title = []
	for key,item in errors.items():
		data.append(item)
		title.append(key)
		# plt.subplot(len(errors), 1, rows)
		# x = numpy.arange(0, len(item), 1)
		# plt.axhline(y=numpy.mean(item), color='r', linestyle = '--', alpha = 0.8, label = "average error",linewidth=3)
		# plt.scatter(x, numpy.array(item), s = 40, c = 'b', marker = 'o', alpha = 0.7, edgecolors='white', label = " error")
		# plt.ylabel('Hellinger Distance')
		# plt.xlabel('Runs (Bias = ' + str(model._bias) + ')')
		# plt.title(key + ' (Data Size = ' + str(model._sample_size) + ', Global epsilon = ' + str(model._epsilon) + ')')
		# plt.legend(loc="best")
		# rows = rows + 1
		# plt.ylim(-0.1,1.0)
		# plt.xlim(0.0,len(item)*1.0)
		# plt.grid()
	fig, ax = plt.subplots()
	bplot = plt.boxplot(data, notch=1, widths=0.4, sym='+', vert=2, whis=1.5,patch_artist=True)
	plt.xlabel("different mechanisms")
	plt.ylabel('Distance Based on L1 Norm Distance')
	#ax.set_xlim(0.5, len(errors) + 0.5)

	plt.xticks(range(1, len(errors)+1),title)
	plt.title('Accuracy / data set:' + str(model._observation_counts) + ", posterior: " + str(model._posterior._alphas) + ", epsilon:" + str(model._epsilon))
	for box in bplot["boxes"]:
		box.set(color='navy', linewidth=1.5)
		box.set(facecolor='lightblue' )
	plt.show()
	# plt.savefig(filename)
	return




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

	while i < len(sorted_scores):
		j = i
		candidates_for_print = []
		candidates_for_classify = []
		while True:
			if (i+1) > len(sorted_scores) or sorted_scores[j][1] != sorted_scores[i][1]:
				break
			candidates_for_print.append(sorted_scores[i][0]._alphas)
			candidates_for_classify.append(sorted_scores[i][0])
			# print sorted_scores[i]
			i += 1
		candidates_classfied_by_steps.append(candidates_for_classify)
		probabilities_exp_by_steps.append(Bayesian_Model._SS_probabilities[j]*(i - j))
		print "Pr[H(BI(x), r) = " + str(-sorted_scores[j][1]) + " ] = " + str(Bayesian_Model._SS_probabilities[j]*(i - j)) + " (r = " + str(candidates_for_print) +")"
   
	# y = numpy.arange(0,4,1)
	laplace_probabilities = {}
	for i in range(len(Bayesian_Model._candidates)):
		r = Bayesian_Model._candidates[i]
		t = 1.0
		# ylist = []
		for j in range(len(r._alphas) - 1):
			a = r._alphas[j] - Bayesian_Model._posterior._alphas[j]
			t = t * 0.5 * (math.exp(- ((abs(a)) if (a >= 0) else (abs(a) - 1)) / (2.0/epsilon)) - math.exp(- ((abs(a) + 1) if (a >= 0) else (abs(a))) / (2.0/epsilon)))
		# 	ylist.append(a)
		# yset = set(ylist)
		laplace_probabilities[r] = t #/ (math.gamma(len(yset)) * (2 ** (len(list(filter(lambda a: a != 0, ylist))))))

	for class_i in candidates_classfied_by_steps:
		pro_i = 0.0
		candidates_for_print = []
		for c in class_i:
			#print laplace_probabilities[c]
			pro_i += laplace_probabilities[c]
			candidates_for_print.append(c._alphas)
		probabilities_lap_by_steps.append(pro_i)
		print "Pr[H(BI(x), r) = " + str(-Bayesian_Model._candidate_scores[class_i[0]]) + " ] = " + str(pro_i) + " (r = " + str(candidates_for_print) +")"


	plt.plot(steps, probabilities_exp_by_steps, 'ro', label=('Exp Mech'))
	# plt.plot(T, approximate_bounds, 'g^', label=('Expmech_SS zApproximate Bound'))
	plt.plot(steps, probabilities_lap_by_steps, 'bs', label=('Laplace Mech'))
	plt.xlabel("c / (steps from correct answer, in form of Hellinger Distance)")
	plt.ylabel("Pr[H(BI(x),r) = c]")
	plt.title("datasize: "+ str(sample_size) + ", x: "+ str(observation) + ", BI(x): beta"+ str(Bayesian_Model._posterior._alphas) + ", epsilon: "+ str(epsilon))
	plt.legend()
	plt.grid()
	plt.show()


def global_epsilon_study(sample_sizes,epsilon,delta,prior):
	epsilons = []
	for n in sample_sizes:
		Bayesian_Model = BayesInferwithDirPrior(prior, n, epsilon, delta)
		Bayesian_Model._set_candidate_scores()
		candidates = Bayesian_Model._candidates
		for i in range(len(candidates)):
			candidates[i]._minus(prior)
		# print candidates
		epsilon_of_n = 0.0
		pair_of_n = []
		for c in candidates:
			temp = deepcopy(c._alphas)
			temp[0] -= 1
			temp[1] += 1
			# print c._alphas, temp
			t = epsilon_study(n,epsilon,delta,prior,c._alphas,temp)
			if epsilon_of_n < t:
				epsilon_of_n = t
				pair_of_n = [c._alphas,temp]

		print " Actually epsilon value:", str(epsilon_of_n), str(pair_of_n)
		epsilons.append(epsilon_of_n)

	plt.figure()
	plt.title(("epsilon study wrt. the data size/ prior: " + str(prior._alphas)))

	plt.plot(sample_sizes,epsilons, 'bo-', label=('Exp Mech'))
	plt.xlabel("Data Size")
	plt.ylabel("maximum epsilon of data size n")
	plt.grid()
	plt.legend(loc='best')
	plt.show()



def epsilon_study(sample_size,epsilon,delta,prior,x1, x2):
	x1_probabilities = epsilon_study_discrete_probabilities(sample_size,epsilon,delta,prior,x1)
	x2_probabilities = epsilon_study_discrete_probabilities(sample_size,epsilon,delta,prior,x2)
	# print x1_probabilities
	accuracy_epsilons = {}
	for key, item in x1_probabilities.items():
		for k,i in x2_probabilities.items():
			if key._alphas == k._alphas:
				accuracy_epsilons[str(key._alphas)] = math.log(item / i)

	sorted_epsilons = sorted(accuracy_epsilons.items(), key=operator.itemgetter(1))
	# print sorted_epsilons
	# print x1, x2
	# if sorted_epsilons[-1][1] > abs(sorted_epsilons[0][1]):
	# 	print sorted_epsilons[-1]
	# else:
	# 	print sorted_epsilons[0]
	return max(sorted_epsilons[-1][1], abs(sorted_epsilons[0][1]))
	
	for key,value in sorted_epsilons:
		print "Pr[ ( M(x1) = " + key + ") / ( M(x2) = " + key + ") ] = exp(" + str(value) + ")"

	y = [value for key, value in sorted_epsilons]

	x = range(len(sorted_epsilons))

	xlabel = [key for key, value in sorted_epsilons]
	plt.figure(figsize=(15,8))
	plt.plot(x, y, 'bs-', label=('Exp Mech'))
	# plt.plot(T, approximate_bounds, 'g^', label=('Expmech_SS zApproximate Bound'))
	plt.xlabel("z / (candiates)")
	plt.ylabel("Pr[(Pr[ ( M(x1) = z) / ( M(x2) = z) ])] = exp(y)")

	plt.title("datasize: "+ str(sample_size) + ", x1: "+ str(x1) + ", x2: "+ str(x2) + ", epsilon: "+ str(epsilon))
	plt.legend()
	plt.xticks(x,xlabel,rotation=70,fontsize=8)
	plt.grid()
	plt.show()


def epsilon_study_discrete_probabilities(sample_size,epsilon,delta,prior,observation):
	Bayesian_Model = BayesInferwithDirPrior(prior, sample_size, epsilon, delta)
	Bayesian_Model._set_observation(observation)

	Bayesian_Model._set_candidate_scores()
	Bayesian_Model._set_LS()
	Bayesian_Model._set_SS()
	# print Bayesian_Model._SS_probabilities
	# print sorted_scores

	probabilities_exp = {}

	for i in range(len(Bayesian_Model._candidates)):
		z = Bayesian_Model._candidates[i]
		probabilities_exp[z] = Bayesian_Model._SS_probabilities[i]
		# print "Pr[ z = " + str(z._alphas) + " ] = " + str(Bayesian_Model._SS_probabilities[i])
	
	return probabilities_exp

def accuracy_VS_epsilon(sample_size,epsilons,delta,prior,observation):
	exp_accuracy = []
	lap_accuracy = []
	exp_accuracy_average = []
	lap_accuracy_average = []

	for epsilon in epsilons:
		Bayesian_Model = BayesInferwithDirPrior(prior, sample_size, epsilon, delta)
		Bayesian_Model._set_observation(observation)
		Bayesian_Model._experiments(150)
		exp_accuracy.append(Bayesian_Model._accuracy[Bayesian_Model._keys[3]])
		lap_accuracy.append(Bayesian_Model._accuracy[Bayesian_Model._keys[0]])
		exp_accuracy_average.append(statistics.mean(Bayesian_Model._accuracy[Bayesian_Model._keys[3]]))
		lap_accuracy_average.append(statistics.mean(Bayesian_Model._accuracy[Bayesian_Model._keys[0]]))
	plt.figure(figsize=(15,8))

	for i in range(len(epsilons)):
		for a in exp_accuracy[i]:
			plt.plot(epsilons[i], a, c = 'r', marker = 'o', markeredgecolor = 'none', alpha = 0.2)
		for b in lap_accuracy[i]:
			plt.plot(epsilons[i], b, c = 'b', marker = 'o', markeredgecolor = 'none', alpha = 0.2)

	plt.plot(epsilons,exp_accuracy_average, 'ro-', label=('Exp Mech (mean)'))
	plt.plot(epsilons,lap_accuracy_average, 'bo-', label=('Laplace Mech (mean)'))

	plt.title("datasize: "+ str(sample_size) + ", x: "+ str(observation))

	plt.ylabel("Accuracy/Hellinger Distance")
	plt.xlabel("epsilon")

	plt.legend(loc="best")
	plt.grid()
	plt.show()

	return

def hellinger_vs_l1norm(base_distribution):
	l1s = numpy.arange(1, 8, 1)
	
	#l1s = []
	hellingers = []
	xstick = []
	for i in l1s:
		# getcontext().prec = 3
		label = deepcopy(base_distribution._alphas)
		label[0] += i
		label[1] -= i
		# label[0] -= Decimal(i) / Decimal(100.0)
		# label[1] += Decimal(i) / Decimal(100.0)
		# label[0] = float(label[0])
		# label[1] = float(label[1])
		xstick.append(label)
		hellingers.append(base_distribution - Dir(label))
		# hellingers.append(Hellinger_Distance_Dir(base_distribution, Dir(label)))

	plt.figure(figsize=(15,10))
	plt.plot(l1s, hellingers, label="hellinger")
	plt.plot(l1s,0.2344 * l1s,label='linear')
	plt.ylabel("Hellinger Distance")
	plt.xlabel("l1 Norm")
	plt.legend()

	plt.xticks(l1s,xstick, rotation=70,fontsize=8)

	plt.show()

def accuracy_VS_datasize(epsilon,delta,prior,observations,mean):
	data = []
	xlabel = []
	for observation in observations:
		Bayesian_Model = BayesInferwithDirPrior(prior, sum(observation), epsilon, delta)
		Bayesian_Model._set_observation(observation)
		Bayesian_Model._experiments(1000)
		data.append(Bayesian_Model._accuracy[Bayesian_Model._keys[3]])
		data.append(Bayesian_Model._accuracy[Bayesian_Model._keys[0]])
		xlabel.append(str(observation) + "/ExpMech")
		xlabel.append(str(observation) + "/Laplace")

	plt.figure(figsize=(12,10))
	bplot = plt.boxplot(data, notch=1, widths=0.4, sym='+', vert=2, whis=1.5,patch_artist=True)
	plt.xlabel("different datasize",fontsize=15)
	plt.ylabel('Accuracy / Hellinegr Distance',fontsize=15)
	#ax.set_xlim(0.5, len(errors) + 0.5)

	plt.xticks(range(0, len(data)),xlabel,rotation=70,fontsize=12)
	plt.title("Accuracy VS. Data Size",fontsize=20)
	print('Accuracy / prior: ' + str(prior._alphas) + ", delta: " + str(delta) + ", epsilon:" + str(epsilon) +  ', mean:' + str(mean))
	for i in range(1, len(bplot["boxes"])/2 + 1):
		box = bplot["boxes"][2 * (i - 1)]
		box.set(color='navy', linewidth=1.5)
		box.set(facecolor='lightblue' )
		box = bplot["boxes"][2 * (i - 1) + 1]
		box.set(color='navy', linewidth=1.5)
		box.set(facecolor='darkkhaki' )
	plt.grid()
	plt.show()

	return

def accuracy_VS_prior(sample_size,epsilon,delta,priors,observation,mean):
	data = []
	xlabel = []
	for prior in priors:
		Bayesian_Model = BayesInferwithDirPrior(prior, sample_size, epsilon, delta)
		Bayesian_Model._set_observation(observation)
		Bayesian_Model._experiments(500)
		data.append(Bayesian_Model._accuracy[Bayesian_Model._keys[3]])
		data.append(Bayesian_Model._accuracy[Bayesian_Model._keys[0]])
		xlabel.append(str(prior._alphas) + "/ExpMech")
		xlabel.append(str(prior._alphas) + "/Laplace")

	plt.figure(figsize=(18,10))
	bplot = plt.boxplot(data, notch=1, widths=0.4, sym='+', vert=2, whis=1.5,patch_artist=True)
	plt.xlabel("different prior distributions",fontsize=15)
	plt.ylabel('Accuracy / Hellinegr Distance',fontsize=15)
	#ax.set_xlim(0.5, len(errors) + 0.5)

	plt.xticks(range(0, len(data)),xlabel,rotation=70,fontsize=12)
	plt.title("Accuracy VS. Prior Distribution",fontsize=20)
	print('Accuracy / observation: ' + str(observation) + ", delta: " + str(delta) + ", epsilon:" + str(epsilon) +  ', mean:' + str(mean))
	for i in range(1, len(bplot["boxes"])/2 + 1):
		box = bplot["boxes"][2 * (i - 1)]
		box.set(color='navy', linewidth=1.5)
		box.set(facecolor='lightblue' )
		box = bplot["boxes"][2 * (i - 1) + 1]
		box.set(color='navy', linewidth=1.5)
		box.set(facecolor='darkkhaki' )
	plt.grid()
	plt.show()
	return

def accuracy_VS_mean(sample_size,epsilon,delta,prior):
	data = []
	xlabel = []
	temp = BayesInferwithDirPrior(prior, sample_size, epsilon, delta)
	temp._set_candidate_scores()
	observations = temp._candidates
	for i in range(len(observations)):
		observations[i]._minus(prior)
	for observation in observations:
		# observation = [int(i * sample_size) for i in mean[:-1]]
		# observation.append(sample_size - sum(observation))
		# print observation
		# print observation._alphas
		Bayesian_Model = BayesInferwithDirPrior(prior, sample_size, epsilon, delta)
		Bayesian_Model._set_observation(observation._alphas)
		Bayesian_Model._experiments(500)
		data.append(Bayesian_Model._accuracy[Bayesian_Model._keys[3]])
		data.append(Bayesian_Model._accuracy[Bayesian_Model._keys[0]])
		xlabel.append(str(observation._alphas) + "/ExpMech")
		xlabel.append(str(observation._alphas) + "/Laplace")

	plt.figure(figsize=(18,10))
	bplot = plt.boxplot(data, notch=1, widths=0.4, sym='+', vert=2, whis=1.5,patch_artist=True)
	plt.xlabel("different observed data sets",fontsize=16)
	plt.ylabel('Accuracy / Hellinegr Distance',fontsize=16)
	#ax.set_xlim(0.5, len(errors) + 0.5)

	plt.xticks(range(1, len(data)+1),xlabel,rotation=70,fontsize=13)
	plt.title("Accuracy VS. Data Variance",fontsize=20)
	print 'Accuracy / data_size: ' + str(sample_size) +  ', prior:' + str(prior._alphas) + ", delta: " + str(delta) + ", epsilon:" + str(epsilon)
	for i in range(1, len(bplot["boxes"])/2 + 1):
		box = bplot["boxes"][2 * (i - 1)]
		box.set(color='navy', linewidth=1.5)
		box.set(facecolor='lightblue' )
		box = bplot["boxes"][2 * (i - 1) + 1]
		box.set(color='navy', linewidth=1.5)
		box.set(facecolor='darkkhaki' )
	plt.grid()
	plt.show()
	return

def accuracy_VS_dimension(sample_sizes, epsilon, delta):
	data = []
	xlabel = []
	for n in sample_sizes:
		for d in range(2,5):
			observation = [n for i in range(d)]
			prior = Dir([1 for i in range(d)])
			Bayesian_Model = BayesInferwithDirPrior(prior, n*d, epsilon, delta)
			Bayesian_Model._set_observation(observation)
			Bayesian_Model._experiments(500)
			data.append(Bayesian_Model._accuracy[Bayesian_Model._keys[3]])
			data.append(Bayesian_Model._accuracy[Bayesian_Model._keys[0]])
			xlabel.append(str(observation) + "/ExpMech")
			xlabel.append(str(observation) + "/Laplace")

	plt.figure(figsize=(18,10))
	bplot = plt.boxplot(data, notch=1, widths=0.4, sym='+', vert=2, whis=1.5,patch_artist=True)
	plt.xlabel("different dimensions",fontsize=15)
	plt.ylabel('Accuracy / Hellinegr Distance',fontsize=15)
	#ax.set_xlim(0.5, len(errors) + 0.5)

	plt.xticks(range(1, len(data)+1),xlabel,rotation=70,fontsize=12)
	plt.title("Accuracy VS. Dimensionality",fontsize=20)
	print('Accuracy / prior: [1,1,...]' + ", delta: " + str(delta) + ", epsilon:" + str(epsilon) +  ', mean: uniform')
	for i in range(1, len(bplot["boxes"])/2 + 1):
		box = bplot["boxes"][2 * (i - 1)]
		box.set(color='navy', linewidth=1.5)
		box.set(facecolor='lightblue' )
		box = bplot["boxes"][2 * (i - 1) + 1]
		box.set(color='navy', linewidth=1.5)
		box.set(facecolor='darkkhaki' )
	plt.grid()
	plt.show()
	return

def accuracy_VS_prior_mean(sample_size,epsilon,delta,priors,observations):
	data = []
	xlabel = []
	for prior in priors:
		for observation in observations:
			Bayesian_Model = BayesInferwithDirPrior(prior, sample_size, epsilon, delta)
			Bayesian_Model._set_observation(observation)
			Bayesian_Model._experiments(300)
			data.append(Bayesian_Model._accuracy[Bayesian_Model._keys[3]])
			data.append(Bayesian_Model._accuracy[Bayesian_Model._keys[0]])
			xlabel.append(str(prior._alphas) + ", data:" + str(observation) + "/ExpMech")
			xlabel.append(str(prior._alphas) + ", data:" + str(observation) + "/Laplace")

	plt.figure(figsize=(18,10))
	bplot = plt.boxplot(data, notch=1, widths=0.4, sym='+', vert=2, whis=1.5,patch_artist=True)
	plt.xlabel("different prior distributions and different data set means")
	plt.ylabel('Accuracy / Hellinegr Distance')
	#ax.set_xlim(0.5, len(errors) + 0.5)

	plt.xticks(range(1, len(data)+1),xlabel,rotation=70)
	plt.title("Accuracy VS. Prior Distribution & Data Variance")
	print ('Accuracy / observation: ' + str(observation) + ", delta: " + str(delta) + ", epsilon:" + str(epsilon) +  ', mean:' + str(mean))
	for i in range(1, len(bplot["boxes"])/2 + 1):
		box = bplot["boxes"][2 * (i - 1)]
		box.set(color='navy', linewidth=1.5)
		box.set(facecolor='lightblue' )
		box = bplot["boxes"][2 * (i - 1) + 1]
		box.set(color='navy', linewidth=1.5)
		box.set(facecolor='darkkhaki' )
	plt.grid()
	plt.show()
	return

if __name__ == "__main__":

	sample_size = 12
	epsilon = 0.8
	delta = 0.00000001
	prior = Dir([40,40])
	x1 = [1,19]
	x2 = [2,18]
	observation = [5,5,5]
	epsilons = numpy.arange(0.1, 2, 0.1)
	sample_sizes = [i for i in range(1,5)]#[300] #[8,12,18,24,30,36,42,44,46,48]#,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80]
	datarange = [1,2,3,5,7,10,15,20,25,30,35,40,45,50,60,70,80,90,100,110,120,130,140,150,160,170]
	observations =[[i*2,i*2] for i in range(950,1000)]
	priors = [Dir([4*i,4*i,4*i]) for i in range(5,20)]
	mean = [int(1.0/len(prior._alphas) * 100)/100.0 for i in range(len(prior._alphas))]
	# print Optimized_Hellinger_Distance_Dir(Dir([250,250]),Dir([249, 249]))
	# accuracy_VS_dimension(sample_sizes, epsilon, delta)
	# accuracy_VS_prior_mean(sample_size,epsilon,delta,priors,observations)
	# means = [[(i/10.0), (1 - i/10.0)] for i in range(1,10)]
	# print means
	# accuracy_VS_prior(sample_size,epsilon,delta,priors,observation,mean)
	# accuracy_VS_mean(sample_size,epsilon,delta,prior)
	accuracy_VS_datasize(epsilon,delta,prior,observations,mean)
	# # hellinger_vs_l1norm(Dir(observation))
	# global_epsilon_study(sample_sizes,epsilon,delta,prior)
	# Dir([1,17]) - Dir([])

	# accuracy_VS_epsilon(sample_size,epsilons,delta,prior,observation)

	
	# epsilon_study(sample_size,epsilon,delta,prior,x1, x2)

	# print math.floor(-0.6)
	# accuracy_study_discrete(sample_size,epsilon,delta,prior,observation)
	# # accuracy_study_exponential_mechanism_SS(sample_size,epsilon,delta,prior,observation)
	# # accuracy_study_laplace(sample_size,epsilon,delta,prior,observation)
	# # Tests the functioning of the module

	# print Dir([10,10]) - Dir([1,19])

	# Bayesian_Model = BayesInferwithDirPrior(prior, sample_size, epsilon, delta)

	# Bayesian_Model._set_observation(observation)

	# Bayesian_Model._experiments(1000)

	# draw_error(Bayesian_Model._accuracy,Bayesian_Model, "order-2-size-30-runs-1000-epsilon-1.2-hellinger-delta000005-observation202020-box.png")

	# draw_error_l1(Bayesian_Model._accuracy_l1,Bayesian_Model, "order-2-size-100-runs-1000-epsilon-08-l1norm-delta000005box.png")
	

