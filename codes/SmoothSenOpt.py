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
from dirichlet import dirichlet
from dpbayesinfer_Betabinomial import BayesInferwithDirPrior

def Hamming_Distance(c1, c2):
	temp = [abs(a - b) for a,b in zip(c1,c2)]
	return sum(temp)/2.0


def smooth_sensitivity_study2(prior, sample_size, epsilon, delta, percentage):
	Bayesian_Model = BayesInferwithDirPrior(prior, sample_size, epsilon,delta)
	observation = [sample_size*i for i in percentage]
	Bayesian_Model._set_observation([sample_size*i for i in percentage])

	Bayesian_Model._set_candidate_scores()
	Bayesian_Model._set_LS_Candidates()
	x = [(c._minus(Bayesian_Model._prior)) for c in Bayesian_Model._candidates]
	beta = 0.001 # math.log(1 - epsilon / (2.0 * math.log(delta / (2.0 * (sample_size)))))
	max_value_list = []
	xstick = [str((r._minus(Bayesian_Model._prior))._alphas) for r in Bayesian_Model._candidates]

	for t in x:
		Bayesian_Model._set_observation(t._alphas)
		# Bayesian_Model._set_candidate_scores()
		# Bayesian_Model._set_LS_Candidates()
		max_value = 0.0
		max_y = ''
		# max_y_list = []
		for r in Bayesian_Model._candidates:
			temp = Bayesian_Model._LS_Candidates[r] * math.exp(- beta * Hamming_Distance(t._alphas, r._alphas))
			if max_value < temp:
				max_y = r._alphas
				max_value = temp

		# for r in Bayesian_Model._candidates:
		# 	temp = Bayesian_Model._LS_Candidates[r] * math.exp(- beta * Hamming_Distance(observation, r._alphas))
		# 	if max_value == temp:
		# 		max_y_list.append(str(r._alphas))
		max_value_list.append(max_value)

		print "when data set x = "+str(t._alphas) + ", smooth sensitivity: S(" + str(t._alphas) + ") = " + str(max_value)

	scatter_plot(y,xstick,'Smooth Sensitivity Given Size' + str(sample_size), )


def smooth_sensitivity_study(prior, sample_size, epsilon, delta, percentage):
	Bayesian_Model = BayesInferwithDirPrior(prior, sample_size, epsilon,delta)
	observation = [sample_size*i for i in percentage]
	Bayesian_Model._set_observation([sample_size*i for i in percentage])

	Bayesian_Model._set_candidate_scores()
	Bayesian_Model._set_LS_Candidates()

	xstick = [str((r._minus(Bayesian_Model._prior))._alphas) for r in Bayesian_Model._candidates]

	beta = 0.00000001

	y=[Bayesian_Model._LS_Candidates[r] * math.exp(- beta * Hamming_Distance(observation, r._alphas)) for r in Bayesian_Model._candidates]

	scatter_plot(y,xstick,'Smooth Sensitivity Study w.r.t. x :' + str([sample_size*i for i in percentage]) + r'; $\epsilon = $' + str(epsilon) + r'; $\delta:$' + str(delta), r"$\Delta_l(H(BI(x'),-))e^{- \gamma *d(x,x')}$")



def scatter_plot(ss,xstick,title,yaxis):
	plt.figure(figsize=(12,6))

	plt.plot(range(len(ss)),ss,'r^')

	# plt.xticks(range(len(ss)),xstick,rotation=70,fontsize=7)
	plt.title(title,fontsize=20)
	plt.xlabel(r"$D$",fontsize=25)	
	plt.ylabel(yaxis,fontsize=15)
	plt.grid()
	plt.legend()
	plt.show()

	return

def ss_exponentiate_component_study(prior, sample_size, epsilon, delta, percentage):
	Bayesian_Model = BayesInferwithDirPrior(prior, sample_size, epsilon,delta)
	observation = [sample_size*i for i in percentage]
	Bayesian_Model._set_observation(observation)

	Bayesian_Model._set_candidate_scores()

	xstick = [str((c._minus(Bayesian_Model._prior))._alphas) for c in Bayesian_Model._candidates]
	# print [c._alphas for c in Bayesian_Model._candidates]

	beta = math.log(1 - epsilon / (2.0 * math.log(delta / (2.0 * (sample_size)))))
	# y=[Hamming_Distance(Bayesian_Model._observation_counts, r._alphas) for r in Bayesian_Model._candidates]
	# print y
	y=[math.exp(- beta * Hamming_Distance(observation, r._alphas)) for r in Bayesian_Model._candidates]


	scatter_plot(y,xstick,'Exponentiate Component of Smooth Sensitivity w.r.t. x :' + str([sample_size*i for i in percentage]) + r'; $\epsilon:$' + str(epsilon) + r'; $\delta:$' + str(delta), r"$e^{- \gamma *d(x,x')}$")

	return

def ss_ls_component_study(prior, sample_size, epsilon, delta, percentage):
	Bayesian_Model = BayesInferwithDirPrior(prior, sample_size, epsilon,delta)
	observation = [sample_size*i for i in percentage]
	Bayesian_Model._set_observation([sample_size*i for i in percentage])

	Bayesian_Model._set_candidate_scores()
	Bayesian_Model._set_LS_Candidates()

	xstick = [str(c._alphas) for c in Bayesian_Model._candidates]
	beta = math.log(1 - epsilon / (2.0 * math.log(delta / (2.0 * (sample_size)))))
	y=[1.0/((1 - Bayesian_Model._LS_Candidates[r] **2)) for r in Bayesian_Model._candidates]

	print abs(y[2] - y[1])
	scatter_plot(y,xstick,"", r"$\frac{1}{LS((\alpha, 100 - \alpha))}$")

	return


if __name__ == "__main__":
	percentage = [0.5,0.5]
	datasize = 1000
	prior = dirichlet([1,1])

	# ss_exponentiate_component_study(prior, datasize, 0.1, 0.00000001, percentage)

	# ss_ls_component_study(prior, datasize, 1.0, 0.00000001, percentage)

	# smooth_sensitivity_study(prior,datasize, 1.0, 0.00000001, percentage)
	
	smooth_sensitivity_study2(prior,datasize,3,0.00000001, percentage)


