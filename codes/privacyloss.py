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






def privacy_loss(sample_sizes,epsilon,delta,prior):
	epsilons = []

	#########################################################################################################################
	#MAX PRIVACY LOSS UNDER SIZE N
	#########################################################################################################################			
	for n in sample_sizes:
		Bayesian_Model = BayesInferwithDirPrior(prior, n, epsilon, delta)
		Bayesian_Model._set_candidate_scores()
		
		candidates = Bayesian_Model._candidates
		#########################################################################################################################
		#ALL POSSIBLE DATA SETS UNDER SIZE N
		for i in range(len(candidates)):
			candidates[i]._minus(prior)

		candidates = [dirichlet([0,n]), dirichlet([1,n-1]),dirichlet([2,n-2]), 
		dirichlet([3, n - 3])]
		#########################################################################################################################
		#MAX PRIVACY LOSS UNDER SIZE N
		#########################################################################################################################			
		privacyloss_of_n = 0.0
		pair_of_n = []
		for C in candidates:
			#########################################################################################################################
			#GET THE ADJACENT DATA SET OF DATA SET C			
			C_adj = deepcopy(C._alphas)
			if C_adj[0] == 0:
				C_adj[0] += 1
				C_adj[1] -= 1
			else:
				C_adj[0] -= 1
				C_adj[1] += 1

			#########################################################################################################################
			#GET THE PRIVACY LOSS UNDER TWO ADJACENT DATA SET C	AND 		
			loss = privacy_loss_x1_x2(n,epsilon,delta,prior,C._alphas,C_adj)
			if privacyloss_of_n < loss:
				privacyloss_of_n = loss
				pair_of_n = [C._alphas,C_adj]

		print " Practical Privacy Loss:", str(privacyloss_of_n), str(pair_of_n)
		epsilons.append(privacyloss_of_n)

	#########################################################################################################################
	#PLOT THE PRIVACY LOSS UNDER DIFFERENT DATA SIZE
	#########################################################################################################################			
	plt.figure()
	plt.title(("PRIVACY LOSS wrt. the data size/ prior: " + str(prior._alphas)))
	plt.plot(sample_sizes,epsilons, 'bo-', label=('Exp Mech with Local Sensitivity'))
	plt.xlabel("Data Size")
	plt.ylabel("maximum privacy loss at data size n")
	plt.grid()
	plt.legend(loc='best')
	plt.show()



def privacy_loss_x1_x2(sample_size,epsilon,delta,prior,x1, x2):

	print "Adjacent data set:", x1, x2
	#########################################################################################################################
	#OBTAIN DECOMPOSED OUTPUTTING PROBABILITIES UNDER TWO ADJACENT DATA SET
	#########################################################################################################################	
	x1_nomalizer, x1_probabilities = decomposed_probability_values(sample_size,epsilon,delta,prior,x1)
	x2_nomalizer, x2_probabilities = decomposed_probability_values(sample_size,epsilon,delta,prior,x2)

	print "NL(x1) =", x1_nomalizer, "NL(x2) =", x2_nomalizer
	print "NL(x2)/NL(x1) = exp(", math.log(x2_nomalizer / x1_nomalizer),")"

	epsilons = {}
	for key, item in x1_probabilities.items():
		i = x2_probabilities[key]
		epsilons[key] = math.log(item / i)
		print "nominator(x1," + key + ") =", item, "; nominator(x2," + key + ") =", i
		print "nominator(x1,r)/nominator(x2,r) = exp(", epsilons[key],")"

	sorted_epsilons = sorted(epsilons.items(), key=operator.itemgetter(1))
	
	
	# return max(sorted_epsilons[-1][1], abs(sorted_epsilons[0][1]))
	#########################################################################################################################
	#OBTAIN OUTPUTTING PROBABILITIES UNDER TWO ADJACENT DATA SET
	#########################################################################################################################	
	x1_probabilities = probability_values(sample_size,epsilon,delta,prior,x1)
	x2_probabilities = probability_values(sample_size,epsilon,delta,prior,x2)

	epsilons = {}
	for key, item in x1_probabilities.items():
		i = x2_probabilities[key]
		epsilons[key] = math.log(item / i)

	sorted_epsilons = sorted(epsilons.items(), key=operator.itemgetter(1))
	
	#########################################################################################################################
	#PLOT THE PRIVACY LOSS FOR 2 ADJACENT DATA SETS
	#########################################################################################################################	
	for key,value in sorted_epsilons:
		print "Pr[ ( M(x1) = " + key + ") / ( M(x2) = " + key + ") ] = exp(" + str(value) + ")"

	return max(sorted_epsilons[-1][1], abs(sorted_epsilons[0][1]))


def decomposed_probability_values(sample_size,epsilon,delta,prior,observation):
	
	Bayesian_Model = BayesInferwithDirPrior(prior, sample_size, epsilon, delta)
	Bayesian_Model._set_observation(observation)

	Bayesian_Model._set_candidate_scores()
	Bayesian_Model._set_local_sensitivities()
	nomalizer = Bayesian_Model._set_up_exp_mech_with_alpha_SS()
	

	probabilities_exp = {}

	for i in range(len(Bayesian_Model._candidates)):
		z = Bayesian_Model._candidates[i]
		probabilities_exp[str(z._alphas)] = nomalizer * Bayesian_Model._alpha_SS_probabilities[i]
	
	return nomalizer, probabilities_exp

def nomalizer_study(datasizes, epsilon, delta, prior):
	loss_in_nomalizer = []
	for datasize in datasizes:
		x1_nomalizer, probabilities = decomposed_probability_values(datasize,epsilon,delta,prior,[datasize-1,1])
		x2_nomalizer, probabilities = decomposed_probability_values(datasize,epsilon,delta,prior,[datasize,0])
		loss_in_nomalizer.append(math.log(x1_nomalizer/x2_nomalizer))

	#########################################################################################################################
	#PLOT THE PRIVACY LOSS UNDER DIFFERENT DATA SIZE
	#########################################################################################################################			
	plt.figure()
	plt.title(("PRIVACY LOSS in Denominator"))
	plt.plot(datasizes,loss_in_nomalizer, 'bo-', label=('Exp Mech with Local Sensitivity'))
	plt.xlabel("Data Size")
	plt.ylabel(r"privacy loss in Denominator ($log(\frac{NL(n-1,1)}{NL(n,0)})$)")
	plt.grid()
	plt.legend(loc='best')
	plt.show()

def probability_values(sample_size,epsilon,delta,prior,observation):
	
	Bayesian_Model = BayesInferwithDirPrior(prior, sample_size, epsilon, delta)
	Bayesian_Model._set_observation(observation)

	Bayesian_Model._set_candidate_scores()
	Bayesian_Model._set_local_sensitivities()
	Bayesian_Model._set_up_exp_mech_with_alpha_SS()
	

	probabilities_exp = {}

	for i in range(len(Bayesian_Model._candidates)):
		z = Bayesian_Model._candidates[i]
		probabilities_exp[str(z._alphas)] = Bayesian_Model._alpha_SS_probabilities[i]
	
	return probabilities_exp



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
	epsilon = 2.0
	delta = 0.00000001
	prior = dirichlet([1,1])
	dataset = [50,50]
	epsilons = numpy.arange(0.1, 2, 0.1)
	datasizes = gen_datasizes((2,12),1) + gen_datasizes((15,50),10) + gen_datasizes((100,500), 100) + gen_datasizes((1000,5000),1000)  + gen_datasizes((10000,50000),5000)# #[300] #[8,12,18,24,30,36,42,44,46,48]#,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80]
	percentage = [0.5,0.5]
	datasets = gen_datasets(percentage, datasizes)
	priors = [dirichlet([1,1])] + gen_priors([5,20], 5, 2) + gen_priors([40,100], 20, 2) + gen_priors([150,300], 50, 2) + gen_priors([400,400], 50, 2)

	# privacy_loss_x1_x2(datasize, epsilon, delta, prior, [3,3],[2,4])
	# privacy_loss(datasizes,epsilon,delta,prior)	
	nomalizer_study(datasizes,epsilon,delta,prior)

