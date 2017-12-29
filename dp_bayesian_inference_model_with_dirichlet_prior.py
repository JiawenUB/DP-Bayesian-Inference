import numpy
import matplotlib.pyplot as plt
from copy import deepcopy
import random
import math
import scipy
from scipy.stats import beta
from fractions import Fraction

def beta_function(alpha, beta):
	f = Fraction(1,1)
	t = (min(alpha, beta))
	k = (max(alpha, beta))
	if t % 1 == 0.0 and k % 1 == 0.0:
		for i in range(1, int(t)):
			f = f * Fraction(i, (i + int(k)))
		f = f * Fraction(1, int(k))
		f = (1.0 * f.numerator) / (f.denominator)
	elif t % 1 == 0.0 and k % 1 != 0.0:
		k = Fraction(int(k * 2), 2)
		f = 1 / k
		t = Fraction((int(t) - 1) * 2, 2)
		while t >= 1:
			f = f * t / (k + t)
			t = t - 1
		f = (1.0 * f.numerator) / (f.denominator)
	elif k % 1 == 0.0 and t % 1 != 0.0:
		t = Fraction(int(t * 2), 2)
		f = 1 / t
		k = Fraction(int(k - 1)*2, 2)
		while k >= 1:
			f = f * k / (k + t) 
			k = k - 1
		f = (1.0 * f.numerator) / (f.denominator)
	else:
		k = Fraction(int(k * 2), 2)
		#print k
		t = Fraction(int(t * 2), 2)
		#print t
		k = k - 1
		s = k + t
		while k > 0:
			#print f,k,s
			f = f * k / s
			k = k - 1
			s = s - 1
		t = t - 1
		while t > 0:
			#print f,t,s
			f = f * t / s
			t = t - 1
			s = s - 1
		f =  math.sqrt(math.pi) * math.sqrt(math.pi) * (1.0 * f.numerator) / (f.denominator)
	return f



def multibeta_function(alphas):
	numerator = 1.0
	denominator = 1.0
	for alpha in alphas:
		numerator = numerator * math.gamma(alpha)
		denominator = denominator + alpha
	return numerator / math.gamma(denominator)


def Hellinger_Distance_Dir(Dir1, Dir2):
	return math.sqrt(1 - multibeta_function((numpy.array(Dir1._alphas) + numpy.array(Dir2._alphas)) / 2.0)/ \
		math.sqrt(multibeta_function(Dir1._alphas) * multibeta_function(Dir2._alphas)))


class Dir(object):
	def __init__(self, alphas):
		self._alphas = alphas
		self._size = len(alphas)

	def __sub__(self, other):
		return Hellinger_Distance_Dir(self, other)

	def __add__(self, other):
		return Dir(list(numpy.array(self._alphas) + numpy.array(other._alphas)))

	def show(self):
		print "Dirichlet"
		print str(self._alphas)

	def _hellinger_sensitivity(self, r):
		LS = 0.0
		temp = deepcopy(r._alphas)
		for i in range(0, r._size):
			temp[i] += 1
			for j in range(i + 1, r._size):
				temp[j] -= 1
				LS = max(LS, abs((r - self) - (self - Dir(temp))))
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
				LS = max(LS, abs((r - self) - (r - Dir(temp))))
				temp[j] += 1
			temp[i] -= 1
		return LS


class BayesInferwithDirPrior(object):
	def __init__(self, prior, sample_size, epsilon):
		self._prior = prior
		self._sample_size = sample_size
		self._epsilon = epsilon
		self._bias = numpy.random.dirichlet(self._prior._alphas)
		self._observation = numpy.random.multinomial(1, self._bias, self._sample_size)
		self._observation_counts = numpy.sum(self._observation, 0)
		self._posterior = Dir(self._observation_counts) + self._prior
		self._laplaced_posterior = self._posterior
		self._randomized_posterior = self._posterior
		self._randomized_observation = deepcopy(self._observation)
		self._exponential_posterior = self._posterior
		self._candidate_scores = {}
		self._candidates = []
		self._GS = 0.0
		self._LS = {}
		self._VS = {}
		self._LS2 = 0.0
		self._candidate_VS_scores = {}
		self._accuracy = {"Laplace Mechanism":[],"Randomize Response":[],"Exponential Mechanism":[]}
		self._average = {"Laplace Mechanism":[],"Randomize Response":[],"Exponential Mechanism":[]}
		self._accuracy_expomech = {"Exponential Mechanism with Local Sensitivity":[],"Laplace Mechanism":[], "Exponential Mechanism with Varying Sensitivity":[], "Exponential Mechanism with Global Sensitivity":[]}		
	
	def _set_bias(self, bias):
		self._bias = bias
		self._update_observation()

	def _update_observation(self):
		self._observation = numpy.random.multinomial(1, self._prior._alphas, self._sample_size)
		self._posterior = Dir(self._observation_counts) + self._prior

	def _set_candidate_scores(self):
		self._set_candidates([], numpy.sum(self._observation))
		# for r in self._candidates:
		# 	print r._alphas
		for r in self._candidates:
			self._candidate_scores[r] = -(self._posterior - r)

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

	def _set_LS(self):
		for r in self._candidates:
			self._LS[r] = r._hellinger_sensitivity(self._posterior)

	def _set_LS_2(self):
		temp = deepcopy(self._posterior._alphas)
		for i in range(0, self._posterior._size):
			temp[i] += 1
			for j in range(i + 1, self._posterior._size):
				temp[j] -= 1
				self._LS2 = max(self._LS2, abs(self._posterior - Dir(temp)))
				temp[j] += 1
			temp[i] -= 1


	def _set_GS(self):
		# for r in self._candidates:
		# 	for j in self._candidates:
		# 		self._GS = max(self._GS, r._hellinger_sensitivity(j))
		self._GS = Dir([1,1,1,2]) - Dir([1,2,1,1])

	def _set_VS(self):
		t = 2 * math.log(len(self._candidates) / 0.9) / self._epsilon
		for r in self._candidates:
			self._candidate_VS_scores[r] = -max([abs((self._candidate_scores[r] + t * self._LS[r] - ((self._candidate_scores[i]) + t * self._LS[i]))/(self._LS[r] + self._LS[i])) for i in self._candidates])
			#self._GS = max(self._GS, self._LS[r])
		print self._candidate_VS_scores

	def _almost_randomize(self):
		return

	def _laplace_noize(self):
		self._laplaced_posterior = Dir([alpha + abs(numpy.random.laplace(0, len(self._prior._alphas) * 1.0/self._epsilon)) for alpha in self._posterior._alphas])

	def _randomize(self):
		return

	def _exponentialize_GS(self):
		probabilities = {}
		nomalizer = 0.0
		for r in self._candidates:
			probabilities[r] = math.exp(self._epsilon * self._candidate_scores[r]/(self._GS))
			nomalizer += probabilities[r]
		outpro = random.random()
		for r in self._candidates:
			if outpro < 0:
				return
			outpro = outpro - probabilities[r]/nomalizer
			self._exponential_posterior = r

	def _exponentialize_LS(self):
		probabilities = {}
		nomalizer = 0.0
		for r in self._candidates:
			probabilities[r] = math.exp(self._epsilon * self._candidate_scores[r]/(self._LS[r]))
			nomalizer += probabilities[r]
		outpro = random.random()
		for r in self._candidates:
			if outpro < 0:
				return
			outpro = outpro - probabilities[r]/nomalizer
			self._exponential_posterior = r

	def _exponentialize_VS(self):
		probabilities = {}
		nomalizer = 0.0
		for r in self._candidates:
			probabilities[r] = math.exp(self._epsilon * self._candidate_VS_scores[r]/(1.0))
			nomalizer += probabilities[r]
		outpro = random.random()
		for r in self._candidates:
			if outpro < 0:
				return
			outpro = outpro - probabilities[r]/nomalizer
			self._exponential_posterior = r


	def _update_expomech(self, times):
		self._set_candidate_scores()
		self._set_LS()
		self._set_GS()
		self._set_LS_2()
		self._set_VS()
		self._show_all()
		for i in range(times):
			self._show_laplaced()
			self._show_exponential()
			#self._exponentialize()
			#self._accuracy_expomech["Exponential Mechanism with Global Sensitivity"].append(self._posterior - self._exponential_posterior)
			self._exponentialize_LS()
			self._accuracy_expomech["Exponential Mechanism with Local Sensitivity"].append(self._posterior - self._exponential_posterior)
			self._exponentialize_VS()
			self._accuracy_expomech["Exponential Mechanism with Varying Sensitivity"].append(self._posterior - self._exponential_posterior)
			self._exponentialize_GS()
			self._accuracy_expomech["Exponential Mechanism with Global Sensitivity"].append(self._posterior - self._exponential_posterior)
			self._laplace_noize()
			self._accuracy_expomech["Laplace Mechanism"].append(self._posterior - self._laplaced_posterior)
			for key,item in self._accuracy.items():
				self._average[key].append(numpy.mean(item))

	def _update_accuracy(self, times):
		for i in range(times):
			self._randomize()
			self._exponentialize()
			self._laplace_noize()
			self._accuracy["Laplace Mechanism"].append(self._posterior - self._laplaced_posterior)
			self._accuracy["Randomize Response"].append(self._posterior - self._randomized_posterior)
			self._accuracy["Exponential Mechanism"].append(self._posterior - self._exponential_posterior)
			for key,item in self._accuracy.items():
				self._average[key].append(numpy.mean(item))

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

	def _show_randomized(self):
		print "The randomized data set is: "
		print self._randomized_observation
		print "The posterior distribution under randomized mechanism is: "
		self._randomized_posterior.show()

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
		self._show_randomized()
		self._show_exponential()
		#self._show_VS()



def draw_distribution(datas, names):
	plt.subplots(nrows=len(datas), ncols=1, figsize=(10,len(datas) * 2.3))
	plt.tight_layout()
	for i in range(len(datas)):
		datas[i] = [round(item * 10 * 2.0)/2.0/10 for item in datas[i]]
		X = list(set(datas[i]))
		X.sort()
		Y = []
		for x in X:
			Y.append(datas[i].count(x)/1000.0)
		plt.subplot(len(datas),1,i+1)
		plt.bar(range(len(Y)), Y, facecolor='g', tick_label = X)
		plt.title(names[i])
		plt.ylim(0,1)
		olt.xlim(0,len(item))
		plt.grid()
	plt.show()

	return

def draw_beta_distribution(datas, names):
	plt.subplots(nrows=len(datas), ncols=1, figsize=(10,len(datas) * 3))
	plt.tight_layout()
	for i in range(len(datas)):
		x = numpy.linspace(scipy.stats.beta.ppf(0.01, datas[i]._alpha, datas[i]._beta),scipy.stats.beta.ppf(0.99, datas[i]._alpha, datas[i]._beta), 100)
		plt.subplot(len(datas),1,i+1)
		plt.plot(x, scipy.stats.beta.pdf(x, datas[i]._alpha, datas[i]._beta),'r-', lw=5, alpha=0.8, label='Beta(' + str(datas[i]._alpha) + "," + str(datas[i]._beta) + ")")
		plt.title(names[i])
		plt.legend()
	plt.show()
	return

def draw_error(errors, model):
	plt.subplots(nrows=len(errors), ncols=1, figsize=(12, len(errors) * 5.0))
	plt.tight_layout(pad=2, h_pad=4, w_pad=2, rect=None)
	rows = 1
	for key,item in errors.items():
		plt.subplot(len(errors), 1, rows)
		x = numpy.arange(0, len(item), 1)
		plt.axhline(y=numpy.mean(item), color='r', linestyle = '--', alpha = 0.8, label = "average error",linewidth=3)
		plt.scatter(x, numpy.array(item), s = 40, c = 'b', marker = 'o', alpha = 0.7, edgecolors='white', label = " error")
		plt.ylabel('Hellinger Distance')
		plt.xlabel('Runs (Bias = ' + str(model._bias) + ', GS = ' + str(model._GS) + ', max LS = ' + str(model._LS2) + ')')
		plt.title(key + ' (Data Size = ' + str(model._sample_size) + ', Global epsilon = ' + str(model._epsilon) + ')')
		plt.legend(loc="best")
		rows = rows + 1
		plt.ylim(-0.1,1.0)
		plt.xlim(0.0,len(item)*1.0)
		plt.grid()
	#plt.show()
	plt.savefig("dirichlet-GS-VS-LS-2.png")
	return

def draw_error_average(averages, model):
	plt.subplots(nrows=len(averages), ncols=1, figsize=(12, len(averages) * 5.0))
	plt.tight_layout(pad=2, h_pad=4, w_pad=2, rect=None)
	rows = 1
	for key,item in averages.items():
		plt.subplot(len(averages), 1, rows)
		x = numpy.arange(2, len(item) + 2, 1)
		plt.plot(x, numpy.array(item), 'r-', lw=2, alpha=0.6)
		plt.ylabel('Hellinger Distance (Bias = ' + str(model._bias) + ')')
		plt.xlabel('Runs')
		plt.title(key + ' (Data Size = ' + str(model._sample_size) + ' epsilon = ' + str(model._epsilon) + ')')
		plt.legend(loc="best")
		rows = rows + 1
	plt.show()
	return


if __name__ == "__main__":
	# Tests the functioning of the module

	sample_size = 120
	epsilon = 0.8
	prior = Dir([2,2])
	Bayesian_Model = BayesInferwithDirPrior(prior, sample_size, epsilon)

	# draw_distribution([list(numpy.random.beta(alpha + Bayesian_Model._get_ones(),\
	# beta + Bayesian_Model._get_zeros(), 1000)), \
	# list(numpy.random.beta(alpha + Bayesian_Model._get_randomized_ones(),\
	# beta + Bayesian_Model._get_randomized_zeros(), 1000)),\
	# list(numpy.random.beta(alpha + Bayesian_Model._get_laplace_ones(),\
	# beta + Bayesian_Model._get_laplace_zeros(), 1000)),\
	# list(numpy.random.beta(alpha + Bayesian_Model._get_exponential_ones(),\
	# beta + Bayesian_Model._get_exponential_zeros(), 1000))],\
	# [" Bayesian Inferred Posterior","Randomized Response Posterior",\
	# "(" + str(epsilon) + ", 0) - DP Posterior Using Laplace Mechanism",\
	# "Exponential Mechanism Posterior"])

	# draw_beta_distribution([Bayesian_Model._posterior,\
	# 	Bayesian_Model._randomized_posterior,\
	# 	Bayesian_Model._laplaced_posterior,\
	# 	Bayesian_Model._exponential_posterior],
	# 	[" Bayesian Inferred Posterior","Randomized Response Posterior",\
	# 	"(" + str(epsilon) + ", 0) - DP Posterior Using Laplace Mechanism",\
	# 	"Exponential Mechanism Posterior"])
	Bayesian_Model._update_expomech(200)

	draw_error(Bayesian_Model._accuracy_expomech,Bayesian_Model)
	# draw_error_average(Bayesian_Model._average)
	# print Beta_Distribution(35,35) - Beta_Distribution(36, 34)
	# print Beta_Distribution(35,35) - Beta_Distribution(36, 35)
	# print Beta_Distribution(35,35) - Beta_Distribution(36, 36)

	#print Beta_Distribution(150,1) - Beta_Distribution(149, 2)
	#print Beta_Distribution(170,1) - Beta_Distribution(169, 2)
	#print Beta_Distribution(51,50) - Beta_Distribution(50, 51)
	#print Beta_Distribution(70,71) - Beta_Distribution(71, 70)
	#print Beta_Distribution(80,81) - Beta_Distribution(81, 80)
	#print Beta_Distribution(500,5) - Beta_Distribution(499, 2)
	#print math.gamma(45.5), 89.0/2 * math.gamma(89.0/2)
	

