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
from dpbayesinfer import BayesInferwithDirPrior

def Hamming_Distance(c1, c2):
	temp = [abs(a - b) for a,b in zip(c1,c2)]
	return sum(temp)/2.0


def smooth_sensitivity_study2(prior, sample_size, epsilon, delta, percentage):
	Bayesian_Model = BayesInferwithDirPrior(prior, sample_size, epsilon,delta)
	Bayesian_Model._set_observation([sample_size*i for i in percentage])

	Bayesian_Model._set_candidate_scores()
	Bayesian_Model._set_LS_Candidates()
	x = [(c._minus(Bayesian_Model._prior)) for c in Bayesian_Model._candidates]
	beta = math.log(1 - epsilon / (2.0 * math.log(delta / (2.0 * (sample_size)))))

	for t in x:
		Bayesian_Model._set_observation(t._alphas)
		# Bayesian_Model._set_candidate_scores()
		# Bayesian_Model._set_LS_Candidates()
		max_value = 0.0
		max_y = ''
		max_y_list = []
		for r in Bayesian_Model._candidates:
			temp = Bayesian_Model._LS_Candidates[r] * math.exp(- beta * Hamming_Distance(Bayesian_Model._observation_counts, [r._alphas[i] - prior._alphas[i] for i in range(prior._size)]))
			if max_value < temp:
				max_y = r._alphas
				max_value = temp

		for r in Bayesian_Model._candidates:
			temp = Bayesian_Model._LS_Candidates[r] * math.exp(- beta * Hamming_Distance(Bayesian_Model._observation_counts, [r._alphas[i] - prior._alphas[i] for i in range(prior._size)]))
			if max_value == temp:
				max_y_list.append(str(r._alphas))

		print "when data set x = "+str(t._alphas) + ", the max value takes at y = " + str(max_y_list)



def smooth_sensitivity_study(prior, sample_size, epsilon, delta, percentage):
	Bayesian_Model = BayesInferwithDirPrior(prior, sample_size, epsilon,delta)
	Bayesian_Model._set_observation([sample_size*i for i in percentage])

	Bayesian_Model._set_candidate_scores()
	Bayesian_Model._set_LS_Candidates()

	xstick = [str((c._minus(Bayesian_Model._prior))._alphas) for c in Bayesian_Model._candidates]
	beta = math.log(1 - epsilon / (2.0 * math.log(delta / (2.0 * (sample_size)))))
	y=[Bayesian_Model._LS_Candidates[r] * math.exp(- beta * Hamming_Distance(Bayesian_Model._observation_counts, [r._alphas[i] - prior._alphas[i] for i in range(prior._size)])) for r in Bayesian_Model._candidates]


	plot_ss(y,xstick,'Smooth Sensitivity Study w.r.t. x :' + str([sample_size*i for i in percentage]), r"$\Delta_l(H(BI(x'),-))e^{- \gamma *d(x,x')}$")


def plot_ss(ss,xstick,title,yaxis):
	plt.figure(figsize=(12,8))
	plt.plot(range(len(ss)),ss,'r^')

	plt.xticks(range(len(ss)),xstick,rotation=70,fontsize=12)
	plt.title(title,fontsize=20)
	plt.xlabel(r"$x'$",fontsize=25)	
	plt.ylabel(yaxis,fontsize=20)
	plt.grid()
	plt.legend()
	plt.show()

	return

def ss_exponentiate_component_study(prior, sample_size, epsilon, delta, percentage):
	Bayesian_Model = BayesInferwithDirPrior(prior, sample_size, epsilon,delta)
	Bayesian_Model._set_observation([sample_size*i for i in percentage])

	Bayesian_Model._set_candidate_scores()

	xstick = [str((c._minus(Bayesian_Model._prior))._alphas) for c in Bayesian_Model._candidates]
	beta = math.log(1 - epsilon / (2.0 * math.log(delta / (2.0 * (sample_size)))))
	y=[math.exp(- beta * Hamming_Distance(Bayesian_Model._observation_counts, [r._alphas[i] - prior._alphas[i] for i in range(prior._size)])) for r in Bayesian_Model._candidates]


	plot_ss(y,xstick,'Exponentiate Component of Smooth Sensitivity w.r.t. x :' + str([sample_size*i for i in percentage]),r"$e^{- \gamma *d(x,x')}$")

	return

def ss_ls_component_study(prior, sample_size, epsilon, delta, percentage):
	Bayesian_Model = BayesInferwithDirPrior(prior, sample_size, epsilon,delta)
	Bayesian_Model._set_observation([sample_size*i for i in percentage])

	Bayesian_Model._set_candidate_scores()
	Bayesian_Model._set_LS_Candidates()

	xstick = [str((c._minus(Bayesian_Model._prior))._alphas) for c in Bayesian_Model._candidates]
	beta = math.log(1 - epsilon / (2.0 * math.log(delta / (2.0 * (sample_size)))))
	y=[Bayesian_Model._LS_Candidates[r] for r in Bayesian_Model._candidates]


	plot_ss(y,xstick,'Local Sensitivity Component of Smooth Sensitivity w.r.t. x :' + str([sample_size*i for i in percentage]),r"$\Delta_l(H(BI(x'),-))}$")

	return


if __name__ == "__main__":
	percentage = [0.02,0.98]
	datasize = 50
	prior = dirichlet([1,1])

	ss_exponentiate_component_study(prior,datasize,0.8,0.00000001,percentage)

	# ss_ls_component_study(prior,datasize,0.8,0.00000001,percentage)
