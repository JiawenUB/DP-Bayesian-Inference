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

def LAPLACE_CDF(interval, scale):
	if interval[0] >= 0.0:
		return (1 - 0.5 * math.exp( (-interval[1]*1.0/scale))) - (1 - 0.5 * math.exp( (-interval[0]/scale)))
	else:
		return (0.5 * math.exp( (interval[1]*1.0/scale))) - (0.5 * math.exp( (interval[0]/scale)))


def lap_distribution_over_candidates(dataobs, prior, eps, sensitivity = 1.0):
	lap_prob = {}

	n = sum(dataobs)

	k = dataobs[0]

	

	############### the expected error when the noises are valid######
	for t in range(- k, n - k + 1):
		c = [k + t, n - k - t]
		lap_prob[str(c)] = 0.5 * LAPLACE_CDF((t, t+1), sensitivity/eps) + 0.5 * LAPLACE_CDF((-t, -t + 1), sensitivity/eps)

	############### the expected error when the noises are not valid######
	lap_prob[str([0,n])] += 0.5 * (0.5 - LAPLACE_CDF((- k, 0.0), sensitivity/eps)) +  0.5 * (0.5 - LAPLACE_CDF((0.0, k + 1), sensitivity/eps))

	lap_prob[str([n,0])] += 0.5 * (0.5 - LAPLACE_CDF((0.0, n - k + 1), sensitivity/eps)) + 0.5 * (0.5 - LAPLACE_CDF((-(n - k), 0.0), sensitivity/eps))
	
	############### the expected error when the noises are not valid######

	print lap_prob

	return lap_prob

def exp_distribution_over_candidates(dataobs, prior, epsilon):
	n = sum(dataobs)
	Bayesian_Model = BayesInferwithDirPrior(prior, n, epsilon)
	Bayesian_Model._set_observation(dataobs)

	Bayesian_Model._set_candidate_scores()
	Bayesian_Model._set_local_sensitivities()
	Bayesian_Model._set_up_exp_mech_with_gamma_SS()

	exp_prob = {}

	for i in range(len(Bayesian_Model._candidates)):
		z = Bayesian_Model._candidates[i]._pointwise_sub(prior)
		exp_prob[str(z._alphas)] = Bayesian_Model._gamma_SS_probabilities[i]
	
	print exp_prob
	
	return exp_prob

def get_steps(dataobs, prior, epsilon):
	n = sum(dataobs)
	Bayesian_Model = BayesInferwithDirPrior(prior, n, epsilon)
	Bayesian_Model._set_observation(dataobs)

	Bayesian_Model._set_candidate_scores()

	cpara = []
	for c in Bayesian_Model._candidates:
		cpara.append(str(c._minus(prior)._alphas))

	return cpara

def list_of_map(keys, maps):
	lists = []
	for m in maps:
		l = []
		for k in keys:
			l.append(m[k])
		lists.append(l)
	return lists


def plot_2d(y,xstick,labels, title):
	plt.figure()
	x = range(len(xstick))
	for i in range(len(y)):
		plt.plot(x,y[i],label=labels[i])
	plt.xlabel(r"different dataset: $[k, n- k]$")
	plt.xticks(x,xstick,rotation=70,fontsize=6)
	plt.title(title)
	plt.legend(loc='best',fontsize=6)
	plt.grid()
	plt.show()


def gen_dataset(v, n):
	return [int(n * i) for i in v]


def gen_datasets(v, n_list):
	return [gen_dataset(v,n) for n in n_list]


def gen_datasizes(r, step):
	return [i*step for i in range(r[0]/step,r[1]/step + 1)]


def gen_priors(r, step, d):
	return [dirichlet([step*i for j in range(d)]) for i in range(r[0]/step,r[1]/step + 1)]


if __name__ == "__main__":

	datasize = 50
	epsilon = 0.1
	delta = 0.00000001
	prior = dirichlet([1,1])
	dataobs = [2,0]
	datasizes = gen_datasizes((50, 2000), 50)

	lap_prob = lap_distribution_over_candidates(dataobs, prior, epsilon, 1.0)
	lap_prob_2 = lap_distribution_over_candidates(dataobs, prior, epsilon, 2.0)
	exp_prob = exp_distribution_over_candidates(dataobs, prior, epsilon)
	steps = get_steps(dataobs, prior, epsilon)

	plot_2d(list_of_map(steps, [lap_prob, lap_prob_2, exp_prob]),
		steps, [r"$\mathsf{LSHist}$", r"$\mathsf{LSDim}$", r"$\mathsf{EHDS}$"], "prob of each candidates with data: "+str(dataobs)+", eps: " + str(epsilon))

