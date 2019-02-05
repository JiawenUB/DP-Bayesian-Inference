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


#########################################################################################################################
###########################Expectation of Error (Accuracy) of Laplace Mechanism##########################################
def mean_error_fix_n(n, prior, epsilon, times, mech):

	Bayesian_Model = BayesInferwithDirPrior(prior, n, epsilon)
	Bayesian_Model._set_observation(gen_dataset_rand(len(prior._alphas), n))
	Bayesian_Model._set_candidate_scores()
	candidates = Bayesian_Model._candidates

	meanerror = []
	xstick = []
	
	for c in candidates:
		accuracy = []
		Bayesian_Model._set_observation(c._alphas)
		for i in range(times):
			if(mech == "lap"):
				Bayesian_Model._laplace_mechanism_no_post(sensitivity = 1.0)
			elif(mech == "lappost"):
				Bayesian_Model._laplace_mechanism(sensitivity = 1.0)
			accuracy.append(Bayesian_Model._posterior - Bayesian_Model._laplaced_posterior)
		meanerror.append(numpy.mean(accuracy))
		xstick.append(str(list(numpy.array(c._alphas) - 1)))

	return meanerror,xstick




#########################################################################################################################
###########################Local Sensitivity Scaled by 1.0 / epsilon##########################################
def ls_scaled_by_eps(n, prior, epsilon):

	Bayesian_Model = BayesInferwithDirPrior(prior, n, epsilon)
	Bayesian_Model._set_observation(gen_dataset_rand(len(prior._alphas), n))
	Bayesian_Model._set_candidate_scores()
	Bayesian_Model._set_local_sensitivities()

	ls = []
	xstick = []
	
	for c in Bayesian_Model._candidates:
		ls.append(Bayesian_Model._LS_Candidates[c] / epsilon)
		xstick.append(c._alphas)

	return ls,xstick


def test(n, prior, epsilon, times):

	meanerror1,xstick = mean_error_fix_n(n, prior, epsilon, times, "lap")

	meanerror2,xstick = mean_error_fix_n(n, prior, epsilon, times, "lappost")

	ls,_ = ls_scaled_by_eps(n, prior, epsilon)


	plot_2d([meanerror1, meanerror2, ls], xstick, 
		[r"Average of $(HD(Beta(k+\mu, n-k+\mu), Beta(k, n-k))$", 
		r"Average of $(HD(Beta(k+\mu, n-k+\mu), Beta(k, n-k))$ with post-processing", 
		r"(LS of $Beta(k, n- k))/ \epsilon$"],
		"With data size " + str(n))
	
	return


def get_ratio(datasizes, prior, epsilon, times):
	r = []

	r_post = []

	for n in datasizes:

		meanerror,xstick = mean_error_fix_n(n, prior, epsilon, times, "lap")

		meanerror_post,xstick = mean_error_fix_n(n, prior, epsilon, times, "lappost")

		ls,_ = ls_scaled_by_eps(n, prior, epsilon)

		ratio = 0.0

		ratio_post = 0.0

		for i in range(len(ls)):

			ratio = max(ratio, meanerror[i]/ls[i])

			ratio_post = max(ratio_post, meanerror_post[i]/ ls[i])

		r.append(ratio)

		r_post.append(ratio_post)

	plot_2d([r, r_post], datasizes, 
		[r"$\max_{k \in [0,n]}\frac{E[HD(Beta(k+\mu, n-k+\mu), Beta(k, n-k)]}{LS(k)/ \epsilon}$", 
		r"$\max_{k \in [0,n]}\frac{E[HD(Beta(k+\mu, n-k+\mu), Beta(k, n-k)]}{LS(k)/ \epsilon}$ with post-processing"],
		"With epsilon " + str(epsilon))
	
	return

def plot_2d(y,xstick,labels, title):
	plt.figure()
	x = range(len(xstick))
	for i in range(len(y)):
		plt.plot(x,y[i],label=labels[i])
	plt.xlabel(r"($[k, n-k]$)")
	plt.xticks(x,xstick,rotation=70,fontsize=6)
	plt.title(title)
	plt.legend(loc='best',fontsize=6)
	plt.grid()
	plt.show()

def plot_2d_numerator(y,x,labels, title):
	plt.figure()

	for i in range(len(y)):
		plt.plot(x,y[i],label=labels[i])
	plt.xlabel(r"($[k, n-k]$)")
	plt.title(title)
	plt.legend(loc='best',fontsize=6)
	plt.grid()
	plt.show()

def gen_dataset(v, n):
	return [int(n * i) for i in v]

def gen_dataset_rand(d, n):
	v = []
	for i in range(d - 1):
		v.append(random.randint(1,10) / 10.0 *(1.0 - sum(v)))
	v.append(1.0 - sum(v))
	return [int(n * i) for i in v]

def gen_datasets(v, n_list):
	return [gen_dataset(v,n) for n in n_list]


def gen_datasizes(r, step):
	return [(r[0] + i*step) for i in range(0,(r[1] - r[0])/step + 1)]

def gen_priors(r, step, d):
	return [dirichlet([step*i for j in range(d)]) for i in range(r[0]/step,r[1]/step + 1)]

def gen_gammas(r, step, scale):
	return [((r[0] + i*step) * scale) for i in range(0,(r[1] - r[0])/step + 1)]

if __name__ == "__main__":

	datasize = 100
	epsilon = 1.0
	delta = 0.00000001
	prior = dirichlet([1,1])
	dataset = [50,50]
	# test(datasize, prior, epsilon, 5000)

	get_ratio([100, 500, 1000, 2000, 3000, 5000], prior, epsilon, 5000)



