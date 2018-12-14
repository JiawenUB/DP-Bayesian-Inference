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



def denumerators(datasizes,datasets,prior, epsilon, delta):
	denumerator = []
	for dataset in datasets:
		Bayesian_Model = BayesInferwithDirPrior(prior, sum(dataset), epsilon, delta)
		Bayesian_Model._set_observation(dataset)
		Bayesian_Model._set_candidate_scores()
		Bayesian_Model._set_local_sensitivities()
		denumerator.append(1.0 / Bayesian_Model._set_up_exp_mech_with_alpha_SS())

	plot_denumerators(datasizes, denumerator)
	return


def plot_denumerators(x,y):
	plt.figure()
	plt.title(("DENUMERATOR"))
	plt.plot(x,y, 'bo-', label=(r'Exp Mech with $\gamma$ Sensitivity'))
	plt.xlabel("Data Size")
	plt.ylabel("denumerator")
	plt.grid()
	plt.legend(loc='best')
	plt.show()
	return

def gen_dataset(v, n):
	return [int(n * i) for i in v]

def gen_datasets(v, n_list):
	return [gen_dataset(v,n) for n in n_list]


def gen_datasizes(r, step):
	return [(r[0] + i*step) for i in range(0,(r[1] - r[0])/step + 1)]

def gen_priors(r, step, d):
	return [dirichlet([step*i for j in range(d)]) for i in range(r[0]/step,r[1]/step + 1)]


if __name__ == "__main__":

	datasize = 20
	epsilon = 1.0
	delta = 0.00000001
	prior = dirichlet([1,1])
	dataset = [50,50]
	epsilons = numpy.arange(0.1, 2, 0.1)
	datasizes = gen_datasizes((2,20),2) + gen_datasizes((25,100),10) + gen_datasizes((100,500), 100)# + gen_datasizes((1000,5000),1000)  + gen_datasizes((10000,50000),10000)
	# datasizes =  gen_datasizes((15,50),10)
	percentage = [0.5,0.5]
	datasets = gen_datasets(percentage, datasizes)
	priors = [dirichlet([1,1])] + gen_priors([5,20], 5, 2) + gen_priors([40,100], 20, 2) + gen_priors([150,300], 50, 2) + gen_priors([400,400], 50, 2)
	denumerators(datasizes, datasets,prior, epsilon, delta)

