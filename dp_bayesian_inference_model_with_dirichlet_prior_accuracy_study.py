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
		temp = alpha - 1
		while temp > 0.0:
			nominators.append(temp)
			temp -=1.0
		if temp < 0.0:
			nominators.append(math.gamma(1 + temp))
	while denominator > 0.0:
		denominators.append(denominator)
		denominator -= 1.0
	if denominator < 0.0:
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
		for i in range(0, self._size):
			temp[i] += 1
			for j in range(i + 1, self._size):
				temp[j] -= 1
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
		print "Calculating Candidates and Scores....."
		start = time.clock()
		self._set_candidates([], numpy.sum(self._observation))
		for r in self._candidates:
			self._candidate_scores[r] = -(self._posterior - r)
		print str(time.clock() - start) + " seconds."

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
		#self._set_LS_Candidates()
		# beta = self._epsilon / (2.0 * len(self._prior._alphas) * (gamma + 1))
		# self._SS = max([self._LS_Candidates[r] * math.exp(- beta * Optimized_Hellinger_Distance_Dir(self._posterior, r)) for r in self._candidates])
		# key1 = "(" + str(self._epsilon / 2.0) + "," + str(beta) + ") Admissible Niose and " + str(beta) + "-Smooth Sensitivity (" + str(self._SS) + ")|" + str(self._epsilon) + "-DP"
		# self._accuracy[key1] = []
		# self._keys.append(key1)
		# print str(time.clock() - start) + "seconds."
		# print "Calculating Smooth Sensitivity with Hamming Distance....."
		# start = time.clock()
		# self._SS_Hamming = max([self._LS_Candidates[r] * math.exp(- beta * Hamming_Distance(self._posterior, r)) for r in self._candidates])
		# key2 = "(" + str(self._epsilon / 2.0) + "," + str(beta) + ") Admissible Niose and " + str(beta) + "-Smooth Sensitivity (" + str(self._SS_Hamming) + ")|" + str(self._epsilon) + "-DP"
		# self._accuracy[key2] = []
		# self._keys.append(key2)
		# print str(time.clock() - start) + "seconds."
		print "Calculating Smooth Sensitivity with Hamming Distance....."
		start = time.clock()
		beta = math.log(1 - self._epsilon / (2.0 * math.log(self._delta / (2.0 * (self._sample_size)))))
		#self._SS_Hamming = max(self._LS, max([self._LS_Candidates[r] * math.exp(- beta * Hamming_Distance(self._posterior, r)) for r in self._candidates]))
		self._SS_Hamming = self._LS
		key3 = "Exponential Mechanism with " + str(beta) + " - Bound Smooth Sensitivity (" + str(self._SS_Hamming) + ")|(" + str(self._epsilon) + "," + str(self._delta) + ")-DP"
		print key3
		key3 = "ExpoMech of SS"
		self._accuracy[key3] = []
		self._accuracy_l1[key3] = []
		self._keys.append(key3)
		print str(time.clock() - start) + "seconds."

		nomalizer = 0.0


		for r in self._candidates:
			temp = math.exp(self._epsilon * self._candidate_scores[r]/(self._SS_Hamming))
			self._SS_probabilities.append(temp)
			nomalizer += temp

		for i in range(len(self._SS_probabilities)):
			self._SS_probabilities[i] = self._SS_probabilities[i]/nomalizer

		return nomalizer
		# beta = 0.0
		# self._SS_Expon = max(self._LS, max([self._LS_Candidates[r] * math.exp(- beta * Hamming_Distance(self._posterior, r)) for r in self._candidates]))
		# key2 = "Exponential Mechanism with " + str(beta) + " - Bound Smooth Sensitivity - " + str(self._SS_Expon) + "| Achieving" + str(self._epsilon) + "-DP"
		# self._accuracy[key2] = []
		# self._keys.append(key2)

	def _set_LS(self):
		self._LS = self._posterior._hellinger_sensitivity(self._posterior)#self._posterior
		key = "Exponential Mechanism with Local Sensitivity - " + str(self._LS) + "| Non Privacy"
		print key
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
		temp = deepcopy(self._prior._alphas)
		temp[0] += 1
		temp[1] -= 1
		self._GS = self._prior - Dir(temp)
		key = "Exponential Mechanism with Global Sensitivity - " + str(self._GS) + "| Achieving" + str(self._epsilon) + "-DP"
		print key
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




	def _set_VS(self):
		t = 2 * math.log(len(self._candidates) / 0.8) / self._epsilon
		print "Calculating Varying Sensitivity Scores....."
		start = time.clock()
		self._set_LS()
		for r in self._candidates:
			self._candidate_VS_scores[r] = -max([((-self._candidate_scores[r] + t * self._LS_Candidates[r] - (-self._candidate_scores[i] + t * self._LS_Candidates[i]))/(self._LS_Candidates[r] + self._LS_Candidates[i])) for i in self._candidates])
		key = "Exponential Mechanism with Varying Sensitivity Scores | Achieving " + str(self._epsilon) + "-DP"
		print key
		key = "ExpoMech of VS"
		self._accuracy[key] = []
		self._accuracy_l1[key] = []
		self._keys.append(key)
		print str(time.clock() - start) + "seconds."

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
	# 	temp = [a + self._SS_Hamming * z /alpha for a in self._posterior._alphas]
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
		self._laplaced_posterior = Dir([alpha + abs(numpy.random.laplace(0, 2.0/self._epsilon)) for alpha in self._posterior._alphas])

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


	def _exponentialize_VS(self):
		probabilities = {}
		for r in self._candidates:
			probabilities[r] = math.exp(self._epsilon * self._candidate_VS_scores[r]/(1.0))
			nomalizer += probabilities[r]
		for r in self._candidates:
			probabilities[r] = probabilities[r]	/ nomalizer	

		self._exponential_posterior = numpy.random.choice(self._candidates, p=probabilities.valuse())
		outpro = random.random()
		for r in self._candidates:
			if outpro < 0:
				return
			outpro = outpro - probabilities[r]/nomalizer
			self._exponential_posterior = r

	def _exponentialize_SS(self):
		probabilities = {}
		nomalizer = 0.0
		self._exponential_posterior = numpy.random.choice(self._candidates, p=self._SS_probabilities)

		# for r in self._candidates:
		# 	probabilities[r] = math.exp(self._epsilon * self._candidate_scores[r]/(self._SS_Hamming))
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
			self._laplace_noize_navie()
			self._accuracy[self._keys[0]].append(Hellinger_Distance_Dir(self._posterior, self._laplaced_posterior))
			self._laplace_noize()
			self._accuracy_l1[self._keys[0]].append(L1_Nrom(self._posterior, self._laplaced_posterior))
			self._exponentialize_GS()
			self._accuracy_l1[self._keys[1]].append(L1_Nrom(self._posterior, self._exponential_posterior))
			self._accuracy[self._keys[1]].append(self._posterior - self._exponential_posterior)
			self._exponentialize_LS()
			self._accuracy[self._keys[2]].append(self._posterior - self._exponential_posterior)
			self._accuracy_l1[self._keys[2]].append(L1_Nrom(self._posterior, self._exponential_posterior))
			# self._Smooth_Sensitivity_Noize()
			# self._accuracy[self._keys[2]].append(self._posterior - self._SS_posterior)
			# self._Smooth_Sensitivity_Noize_Hamming()
			# self._accuracy[self._keys[3]].append(self._posterior - self._SS_posterior)
			# self._Smooth_Sensitivity_Laplace_Noize()
			# self._accuracy[self._keys[4]].append(self._posterior - self._SS_posterior)
			self._exponentialize_SS()
			self._accuracy[self._keys[3]].append(self._posterior - self._exponential_posterior)
			self._accuracy_l1[self._keys[3]].append(L1_Nrom(self._posterior, self._exponential_posterior))


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
	plt.title('Accuracy')
	for box in bplot["boxes"]:
		box.set(color='navy', linewidth=1.5)
		box.set(facecolor='lightblue' )
    #plt.show()
	plt.savefig(filename)
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
	plt.title('Accuracy')
	for box in bplot["boxes"]:
		box.set(color='navy', linewidth=1.5)
		box.set(facecolor='lightblue' )
    #plt.show()
	plt.savefig(filename)
	return 

if __name__ == "__main__":
	# Tests the functioning of the module

	sample_size = 498
	epsilon = 0.8
	delta = 0.00005
	prior = Dir([1,1])
	Bayesian_Model = BayesInferwithDirPrior(prior, sample_size, epsilon, delta)
	Bayesian_Model._set_observation([249,249])

	Bayesian_Model._set_candidate_scores()
	Bayesian_Model._set_LS()
	nomalizer = Bayesian_Model._set_SS()	
	c = numpy.arange(0.2,1.0,0.01)

	print nomalizer

	accurate_bounds = []

	approximate_bounds = []

	for bound in c:

		nominator = Bayesian_Model._get_accuracy_bound(bound,Bayesian_Model._SS_Hamming)

		t = nominator/nomalizer

		accurate_bounds.append(t)

		print t

		t = Bayesian_Model._get_approximation_accuracy_bound(bound,Bayesian_Model._SS_Hamming)/nomalizer

		approximate_bounds.append(t)

		print t
	plt.plot(c,accurate_bounds, 'ro', label=('Accurate Bound')
	)
	plt.plot(c, approximate_bounds, 'g^', label=('Approximate Bound'))
	plt.xlabel("c")
	plt.ylabel("Pr[H(BI(x),r) > c]")
	plt.title("datasize: 498, x: (249, 249), BI(x): beta(250,250), epsilon:0.8")
	plt.legend()
	plt.show()
	# Bayesian_Model._experiments(1000)

	# draw_error(Bayesian_Model._accuracy,Bayesian_Model, "order-2-size-100-runs-1000-epsilon-08-hellinger-delta000005-box.png")

	# draw_error_l1(Bayesian_Model._accuracy_l1,Bayesian_Model, "order-2-size-100-runs-1000-epsilon-08-l1norm-delta000005box.png")
	

