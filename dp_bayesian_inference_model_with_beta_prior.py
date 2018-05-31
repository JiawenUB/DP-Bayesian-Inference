import numpy
import matplotlib.pyplot as plt
from copy import deepcopy
import random
import math
import scipy
from scipy.stats import beta
from fractions import Fraction
import time

def beta_function(alpha, beta):
	f = Fraction(1,1)
	t = (min(alpha, beta))
	k = (max(alpha, beta))
	if t % 1 == 0.0 and k % 1 == 0.0:
		for i in range(1, int(t)):
			f = f * Fraction(i, (i + int(k)))
		f = f * Fraction(1, int(k))
		f = (1.0 * f.numerator) / (f.denominator)
	# else:
	# 	f = math.gamma(alpha) * math.gamma(beta) / math.gamma(alpha + beta)
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

def beta_function_naive(alpha, beta):
 	return 1.0 * math.gamma(alpha) * math.gamma(beta)/math.gamma(alpha + beta)

def Varying_Sensitivity(sensitivities, r, hellingers, t):
	score = 0.0
	for i in range(len(sensitivities)):
		score = max(score, (hellingers[r] + t * sensitivities[r] - hellingers[i] - t * sensitivities[i])/(sensitivities[i] + sensitivities[r]))
	return score

def global_sensitivities(a, b, r):
	sensitivities = []
	for i in range(r):
		sensitivities.append(Beta_Distribution(a + i + 1, b + r - i - 1 ) - Beta_Distribution(a + i, b + r - i))
	return sensitivities


class Beta_Distribution(object):
	def __init__(self, alpha, beta):
		self._alpha = alpha
		self._beta = beta

	def __sub__(self, other):
		b = beta_function(self._alpha, self._beta)
		a = beta_function((self._alpha + other._alpha)/2.0, (self._beta + other._beta)/2.0)
		c = beta_function(other._alpha, other._beta)
		return math.sqrt(1 - ( a / math.sqrt(b * c)))

	def show(self):
		print "Beta(" + str(self._alpha) + ", " + str(self._beta) + ")"


class BayesInferwithBetaPrior(object):

	def __init__(self, prior, sample_size, epsilon):

		self._prior = prior
		self._sample_size = sample_size
		self._epsilon = epsilon
		self._bias = numpy.random.beta(self._prior._alpha, self._prior._beta)
		self._observation = numpy.random.binomial(1, self._bias, self._sample_size)
		self._posterior = Beta_Distribution(prior._alpha + numpy.count_nonzero(self._observation == 1),\
			prior._beta + numpy.count_nonzero(self._observation == 0))
		self._laplaced_posterior = self._posterior
		self._randomized_posterior = self._posterior
		self._randomized_observation = deepcopy(self._observation)
		self._exponential_posterior = self._posterior
		self._expomech_sensitivities = global_sensitivities(self._prior._alpha, self._prior._beta, self._sample_size)
		self._expomech_SS = 0.0
		self._scores = {}
		self._candidate_scores = {}
		self._candidates = []
		self._GS = 0.0
		self._candidates_LS = {}
		self._VS = {}
		self._SS = {}
		self._LS = 0.0
		self._SS_Hamming = 0.0
		self._keys = ["Laplace Mechanism | Achieving" + str(self._epsilon) + "-DP"]
		self._accuracy = {self._keys[0]:[]}

	def _set_bias(self, bias):
		self._bias = bias
		self._update_observation()

	def _update_observation(self):
		self._observation = numpy.random.binomial(1, self._bias, self._sample_size)
		self._posterior = Beta_Distribution(prior._alpha + numpy.count_nonzero(self._observation == 1),\
			prior._beta + numpy.count_nonzero(self._observation == 0))

	def _set_candidate_scores(self):
		print "Calculating Candidates and Scores....."
		start = time.clock()
		self._set_candidates([], numpy.sum(self._observation))
		for r in self._candidates:
			self._candidate_scores[r] = -(self._posterior - r)
		print str(time.clock() - start) + " seconds."

	def _set_candidates(self, current, rest):
		if len(current) == len(self._prior._alpha) - 1:
			current.append(rest)
			self._candidates.append(Dir(deepcopy(current)) + self._prior)
			current.pop()
			return
		for i in range(0, rest + 1):
			current.append(i)
			self._set_candidates(current, rest - i)
			current.pop()

	def _set_candidates_LS(self):
		for r in self._candidates:
			self._candidates_LS[r] = r._hellinger_sensitivity(r)

	def _set_SS(self):
		self._set_LS_Candidates()
		# print "Calculating Smooth Sensitivity with Hamming Distance....."
		# start = time.clock()
		# self._SS_Hamming = max([self._LS_Candidates[r] * math.exp(- beta * Hamming_Distance(self._posterior, r)) for r in self._candidates])
		# key2 = "(" + str(self._epsilon / 2.0) + "," + str(beta) + ") Admissible Niose and " + str(beta) + "-Smooth Sensitivity (" + str(self._SS_Hamming) + ")|" + str(self._epsilon) + "-DP"
		# self._accuracy[key2] = []
		# self._keys.append(key2)
		# print str(time.clock() - start) + "seconds."
		print "Calculating Smooth Sensitivity with Hamming Distance....."
		start = time.clock()
		beta = math.log(1 - self._epsilon / (2.0 * math.log(self._delta / (2 (n + 1)))))
		self._SS_Hamming = max([self._LS_Candidates[r] * math.exp(- beta * Hamming_Distance(self._posterior, r)) for r in self._candidates])
		key3 = "(" + str(self._epsilon / 2.0) + "," + str(beta) + ") Admissible Niose and " + str(beta) + "-Smooth Sensitivity (" + str(self._SS_Laplace) + ")|(" + str(self._epsilon) + "," + str(self._delta) + ")-DP"
		self._accuracy[key3] = []
		self._keys.append(key3)
		print str(time.clock() - start) + "seconds."	


	def _set_LS(self):
		self._LS = self._posterior._hellinger_sensitivity(self._posterior)
		key = "Exponential Mechanism with Local Sensitivity - " + str(self._LS) + "| Non Privacy"
		self._accuracy[key] = []
		self._keys.append(key)		

	def _set_GS(self):
		temp = deepcopy(self._prior._alphas)
		temp[0] += 1
		temp[1] -= 1
		self._GS = self._prior - Dir(temp)
		key = "Exponential Mechanism with Global Sensitivity - " + str(self._GS) + "| Achieving" + str(self._epsilon) + "-DP"
		self._accuracy[key] = []
		self._keys.append(key)

	def _laplace_constrained_mle(self):
		a = self._posterior._alpha + round(numpy.random.laplace(0, 2.0/self._epsilon)) - self._prior._alpha
		b = self._posterior._beta + round(numpy.random.laplace(0, 2.0/self._epsilon)) - self._prior._beta
		a = a + int(self._sample_size - a - b) / 2
		b = b + int(self._sample_size - a - b) / 2
		if a + b < int(self._sample_size):
			a = a + 1
		elif a + b >int(self._sample_size):
			a = a - 1
		self._laplaced_posterior = Beta_Distribution(int(self._prior._alpha + a), int(self._prior._beta + b))


	def _laplace_constrained_mindistance(self):
		a = self._posterior._alpha + round(numpy.random.laplace(0, 2.0/self._epsilon)) 
		b = self._posterior._beta + round(numpy.random.laplace(0, 2.0/self._epsilon))
		c = int(self._sample_size - a - b + self._prior._alpha + self._prior._beta)
		#print a,b,c
		mindistance = 100000
		x = y = 0
		if c >= 0:
			for i in range(c + 1):
				if Beta_Distribution(a + i, b + c - i) - Beta_Distribution(a, b) < mindistance:
					mindistance = Beta_Distribution(a + i, b + c - i) - Beta_Distribution(a, b)
					x = a + i
					y = b + c - i
		else:
			c = 0 - c
			for i in range(c + 1):
				if Beta_Distribution(a - i, b - c + i) - Beta_Distribution(a, b) < mindistance:
					mindistance = Beta_Distribution(a - i, b - c + i) - Beta_Distribution(a, b)
					x = a - i
					y = b - c + i			

		self._laplaced_posterior = Beta_Distribution(int(x), int(y))

	def _laplace_noize(self):
		self._laplaced_posterior = Beta_Distribution(self._posterior._alpha + numpy.random.laplace(0, 2.0/self._epsilon),\
			self._posterior._beta + numpy.random.laplace(0, 2.0/self._epsilon))

	def _laplace_noize_rounded(self):
		self._laplaced_posterior = Beta_Distribution(self._posterior._alpha + round(numpy.random.laplace(0, 2.0/self._epsilon)),\
			self._posterior._beta + round(numpy.random.laplace(0, 2.0/self._epsilon)))


	def _almost_randomize(self):
		for i in range(self._sample_size):
			if random.randint(0,1) == 1:
				self._randomized_observation[i] = random.randint(0,1)
			else:
				self._randomized_observation[i] = self._observation[i]
		self._randomized_posterior = Beta_Distribution(self._prior._alpha + numpy.count_nonzero(self._randomized_observation == 1),\
			self._prior._beta + numpy.count_nonzero(self._randomized_observation == 0))

	def _randomize(self):
		for i in range(self._sample_size):
			if random.random() >= math.exp(self._epsilon)/(1.0 + math.exp(self._epsilon)):
				self._randomized_observation[i] = 1 - self._observation[i]
			else:
				self._randomized_observation[i] = self._observation[i]
		self._randomized_posterior = Beta_Distribution(self._prior._alpha +  numpy.count_nonzero(self._randomized_observation == 1),\
			self._prior._beta + numpy.count_nonzero(self._randomized_observation == 0))	

	def _exponentialize(self):
		R = range(self._sample_size - 1)
		scores = {}
		sum_score = 0.0
		delta = math.sqrt( (1 - math.pi/4.0))
		for r in R:
			scores[r] = math.exp(self._epsilon * self._scores(r)/(2 * delta))
			sum_score = sum_score + scores[r]
		outpro = random.random()
		# print scores
		self._exponential_ones = R[0]
		for r in R:
			if outpro < 0:
				return
			outpro = outpro - scores[r]/sum_score
			self._exponential_posterior = Beta_Distribution(self._prior._alpha + r, self._prior._beta + self._sample_size - r)
		return

	def _set_scores(self):
		R = range(self._sample_size - 1)
		for r in R:
			self._scores[r] = -(self._posterior - Beta_Distribution(self._prior._alpha + r, self._prior._beta + self._sample_size - r))


	def _exponentialize_LS(self):
		R = range(self._sample_size - 1)# [(j + 1, i + 1) for i in range(self._sample_size) for j in range(self._sample_size)]
		scores = {}
		sum_score = 0.0
		#delta = math.sqrt( (1 - math.pi/4.0))
		delta = self._posterior - Beta_Distribution(self._posterior._alpha + 1,self._posterior._beta - 1)
		for r in R:
			scores[r] = math.exp(self._epsilon * self._scores[r]/(delta))
			# print scores(r)
			sum_score = sum_score + scores[r]
		outpro = random.random()
		# print scores
		self._exponential_ones = R[0]
		for r in R:
			if outpro < 0:
				return
			outpro = outpro - scores[r]/sum_score
			self._exponential_posterior = Beta_Distribution(self._prior._alpha + r, self._prior._beta + self._sample_size - r)
		return


	def _exponentialize_VS(self):
		R = range(self._sample_size)# [(j + 1, i + 1) for i in range(self._sample_size) for j in range(self._sample_size)]
		def hellinger(r):
			return - (self._posterior - Beta_Distribution(self._prior._alpha + r, self._prior._beta + self._sample_size - r))
		
		hellingers = []
		
		for r in R:
			hellingers.append(hellinger(r))

		def utility(r):
			return Varying_Sensitivity(self._expomech_sensitivities, r, hellingers, 1)

		scores = {}
		sum_score = 0.0
		#delta = math.sqrt( (1 - math.pi/4.0))
		delta = 1
		for r in R:
			scores[r] = math.exp(self._epsilon * utility(r)/(2 * delta))
			# print utility(r)
			sum_score = sum_score + scores[r]
		outpro = random.random()
		# print scores
		self._exponential_ones = R[0]
		for r in R:
			if outpro < 0:
				return
			outpro = outpro - scores[r]/sum_score
			self._exponential_posterior = Beta_Distribution(self._prior._alpha + r, self._prior._beta + self._sample_size - r)
		return
		
	def _exponentialize_SS(self):
		R = range(self._sample_size + 1)# [(j + 1, i + 1) for i in range(self._sample_size) for j in range(self._sample_size)]
		scores = {}
		sum_score = 0.0
		#delta = math.sqrt( (1 - math.pi/4.0))
		print self._posterior._alpha,self._posterior._beta
		delta = self._posterior - Beta_Distribution(self._posterior._alpha - 1,self._posterior._beta + 1)
		print delta
		for r in R:
			scores[r] = math.exp(self._epsilon * self._scores[r]/(delta))
			# print scores(r)
			sum_score = sum_score + scores[r]
		outpro = random.random()
		# print scores
		self._exponential_ones = R[0]
		for r in R:
			print scores[r]/sum_score
		for r in R:
			if outpro < 0:
				return
			outpro = outpro - scores[r]/sum_score
			self._exponential_posterior = Beta_Distribution(self._prior._alpha + r, self._prior._beta + self._sample_size - r)	
		return

	def _experiments(self, times):
		self._set_candidate_scores()
		self._set_GS()
		self._set_SS()
		self._set_LS()
		self._show_all()
		for i in range(times):
			self._laplace_noize_mle()
			self._accuracy[self._keys[0]].append(self._posterior - self._laplaced_posterior)
			self._exponentialize_GS()
			# self._accuracy[self._keys[1]].append(self._posterior - self._exponential_posterior)
			# self._Smooth_Sensitivity_Noize()
			self._accuracy[self._keys[1]].append(self._posterior - self._SS_posterior)
			self._Smooth_Sensitivity_Noize_Hamming()
			self._accuracy[self._keys[2]].append(self._posterior - self._SS_posterior)
			self._Smooth_Sensitivity_Laplace_Noize()
			self._accuracy[self._keys[3]].append(self._posterior - self._SS_posterior)
			self._exponentialize_LS()
			self._accuracy[self._keys[4]].append(self._posterior - self._exponential_posterior)




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
		plt.xlabel('Runs (Bias = ' + str(model._bias) + ', GS = ' + str(max(model._expomech_sensitivities)) + ', LS = ' + str(model._posterior - Beta_Distribution(model._posterior._alpha + 1,model._posterior._beta - 1)) + ')')
		plt.title(key + ' (Data Size = ' + str(model._sample_size) + ' epsilon = ' + str(model._epsilon) + ')')
		plt.legend(loc="best")
		rows = rows + 1
		plt.ylim(-0.1,0.4)
		plt.xlim(0.0,len(item)*1.0)
		plt.grid()
	plt.show()
	return



if __name__ == "__main__":
	# Tests the functioning of the module

	sample_size = 200
	epsilon = 1
	prior = Beta_Distribution(4, 4)
	Bayesian_Model = BayesInferwithBetaPrior(prior, sample_size, epsilon)

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
	Bayesian_Model._set_bias(0.4)
	#Bayesian_Model._experiments(200)

	#draw_error(Bayesian_Model._accuracy,Bayesian_Model)
	# draw_error_average(Bayesian_Model._average)
	print Beta_Distribution(100,100) - Beta_Distribution(99, 101)
	f = Beta_Distribution(100,100) - Beta_Distribution(99, 101)
	# print Beta_Distribution(35,35) - Beta_Distribution(36, 35)
	# print Beta_Distribution(35,35) - Beta_Distribution(36, 36)

	#print Beta_Distribution(150,1) - Beta_Distribution(149, 2)
	#print Beta_Distribution(170,1) - Beta_Distribution(169, 2)
	#print Beta_Distribution(51,50) - Beta_Distribution(50, 51)
	#print Beta_Distribution(70,71) - Beta_Distribution(71, 70)
	#print Beta_Distribution(80,81) - Beta_Distribution(81, 80)
	#print Beta_Distribution(500,5) - Beta_Distribution(499, 2)
	#print math.gamma(45.5), 89.0/2 * math.gamma(89.0/2)
	a = beta_function_naive(100 + math.log(1./0.05) * math.sqrt(1 - math.pi/4)/2.0, 100 - math.log(1./0.05) * math.sqrt(1 - math.pi/4)/2.0)
	
	b = beta_function_naive(100,100)

	c = beta_function_naive(100 + math.log(1./0.05) * math.sqrt(1 - math.pi/4), 100 - math.log(1./0.05) * math.sqrt(1 - math.pi/4))

	d = -math.sqrt(1 - (a/math.sqrt(b * c)))

	e = 199.0 * math.exp(0.8 * d / (2 * f))

	print d,e
	

