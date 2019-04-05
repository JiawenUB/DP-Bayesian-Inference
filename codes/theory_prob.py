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


def lap_distribution_over_candidates(dataobs, prior, eps):
	lap_prob = {}

	n = sum(dataobs)

	k = dataobs[0]

	candidates = []
	sensitivity = 1.0

	############### the expected error when the noises are valid######
	for t in range(- k, n - k + 1):
		c = dirichlet([k + t, n - k - t])
		candidates.append(c)
		lap_prob[c] = 0.5 * LAPLACE_CDF((t, t+1), sensitivity/eps) + 0.5 * LAPLACE_CDF((-t, -t + 1), sensitivity/eps)

	############### the expected error when the noises are not valid######
	lap_prob[candidates[0]] += 0.5 * (0.5 - LAPLACE_CDF((- k, 0.0), sensitivity/eps)) +  0.5 * (0.5 - LAPLACE_CDF((0.0, k + 1), sensitivity/eps))

	lap_prob[candidates[n]] += 0.5 * (0.5 - LAPLACE_CDF((0.0, n - k + 1), sensitivity/eps)) + 0.5 * (0.5 - LAPLACE_CDF((-(n - k), 0.0), sensitivity/eps))
	
	############### the expected error when the noises are not valid######
	for c,prob in lap_prob.items():
		print c._alphas, prob
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
		print z._alphas, Bayesian_Model._gamma_SS_probabilities[i]
	return exp_prob

if __name__ == "__main__":

	datasize = 50
	epsilon = 1.0
	delta = 0.00000001
	prior = dirichlet([1,1])
	dataset = [50,50]
	datasizes = gen_datasizes((50, 2000), 50)

	# get_separatevalue(datasize, prior, epsilon, 500)

	# get_ratio(datasizes, prior, epsilon, 5000)
	lap_prob = lap_distribution_over_candidates([0,3], prior, epsilon)
	exp_prob = exp_distribution_over_candidates([0,3], prior, epsilon)
