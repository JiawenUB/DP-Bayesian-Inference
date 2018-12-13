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
#OBTAIN OUTPUTTING PROBABILITIES OF ONE DATA SET
#########################################################################################################################	
def probability_values(datasize,epsilon,delta,prior,observation):
	
	Bayesian_Model = BayesInferwithDirPrior(prior, datasize, epsilon, delta)
	Bayesian_Model._set_observation(observation)

	Bayesian_Model._set_candidate_scores()
	Bayesian_Model._set_local_sensitivities()
	Bayesian_Model._set_up_exp_mech_with_alpha_SS()
	

	probabilities_exp = {}

	for i in range(len(Bayesian_Model._candidates)):
		z = Bayesian_Model._candidates[i]
		probabilities_exp[str(z._alphas)] = Bayesian_Model._alpha_SS_probabilities[i]
	
	return probabilities_exp

#########################################################################################################################
#OBTAIN DECOMPOSED OUTPUTTING PROBABILITIES OF ONE DATA SET
#########################################################################################################################	
def decomposed_probability_values(datasize,epsilon,delta,prior,observation):
	
	Bayesian_Model = BayesInferwithDirPrior(prior, datasize, epsilon, delta)
	Bayesian_Model._set_observation(observation)

	Bayesian_Model._set_candidate_scores()
	Bayesian_Model._set_local_sensitivities()
	nomalizer = Bayesian_Model._set_up_exp_mech_with_alpha_SS()
	
	probabilities_exp = {}

	for i in range(len(Bayesian_Model._candidates)):
		z = Bayesian_Model._candidates[i]
		probabilities_exp[str(z._alphas)] = nomalizer * Bayesian_Model._alpha_SS_probabilities[i]
	
	return nomalizer, probabilities_exp

#########################################################################################################################
#OBTAIN THE OUTPUTTING PROBABILITIES OF A PAIR OF ADJACENT DATA SET
#########################################################################################################################	
def privacy_loss_one_pair(datasize,epsilon,delta,prior, x1, x2):

	x1_probabilities = probability_values(datasize,epsilon,delta,prior,x1)
	x2_probabilities = probability_values(datasize,epsilon,delta,prior,x2)

	epsilons = {}
	for key, item in x1_probabilities.items():
		i = x2_probabilities[key]
		epsilons[key] = abs(math.log(item / i))

	return sorted(epsilons.items(), key=operator.itemgetter(1))


#########################################################################################################################
#OBTAIN DECOMPOSED OUTPUTTING PROBABILITIES OF A PAIR OF ADJACENT DATA SET - NUMERATOR
#########################################################################################################################	
def numerator_privacy_loss_one_pair(datasize,epsilon,delta,prior, x1, x2):

	x1_nomalizer, x1_probabilities = decomposed_probability_values(datasize,epsilon,delta,prior,x1)
	x2_nomalizer, x2_probabilities = decomposed_probability_values(datasize,epsilon,delta,prior,x2)

	epsilons = {}
	for key, item in x1_probabilities.items():
		i = x2_probabilities[key]
		epsilons[key] = abs(math.log(item / i))
		# print "numerator", key, epsilons[key]

	return sorted(epsilons.items(), key=operator.itemgetter(1))

#########################################################################################################################
#OBTAIN DECOMPOSED OUTPUTTING PROBABILITIES OF A PAIR OF ADJACENT DATA SET - NUMERATOR
#########################################################################################################################	
def denumerator_privacy_loss_one_pair(datasize,epsilon,delta,prior, x1, x2):

	print "Adjacent data set:", x1, x2, "denumerator"

	x1_nomalizer, x1_probabilities = decomposed_probability_values(datasize,epsilon,delta,prior,x1)
	x2_nomalizer, x2_probabilities = decomposed_probability_values(datasize,epsilon,delta,prior,x2)
	print "NL(x1) =", x1_nomalizer, "NL(x2) =", x2_nomalizer
	print "NL(x2)/NL(x1) = exp(", math.log(x2_nomalizer / x1_nomalizer),")"

	return (math.log(x2_nomalizer / x1_nomalizer))



#########################################################################################################################
#PLOT THE PRIVACY LOSS UNDER DIFFERENT DATA SIZE
#########################################################################################################################			
def plot_privacyloss(x, y, label):
	plt.figure()
	plt.title(("PRIVACY LOSS " + label))
	plt.plot(x,y, 'bo-', label=(r'Exp Mech with $\gamma$ Sensitivity'))
	plt.xlabel("Data Size")
	plt.ylabel(r"privacy loss " + label)
	plt.grid()
	plt.legend(loc='best')
	plt.show()


#########################################################################################################################
# DECOMPOSED PRIVACY UNDER DIFFERENET DATA SIZE - DENUMERATOR
#########################################################################################################################	
def privacy_loss_in_denumerator(datasizes, epsilon, delta, prior):
	loss_in_nomalizer = []
	for datasize in datasizes:
		loss_in_nomalizer.append(denumerator_privacy_loss_one_pair(datasize,epsilon,delta,prior,[datasize-1,1],[datasize,0]))

	plot_privacyloss(datasizes, loss_in_nomalizer, "in denumerator")

#########################################################################################################################
# DECOMPOSED PRIVACY UNDER DIFFERENET DATA SIZE - NUMERATOR
#########################################################################################################################	
def privacy_loss_in_numerator(datasizes, epsilon, delta, prior):
	loss_in_numerator = []
	for datasize in datasizes:
		loss_max_pair = (0.0,0.0)
		candidates = [[datasize-2,2],[datasize-1,1],[datasize, 0],[0,datasize], [1, datasize - 1], [2, datasize - 2]]

		for C in candidates:
		#########################################################################################################################
		#GET THE ADJACENT DATA SET OF DATA SET C			
			C_adj = get_adjacent_set(C)
				#########################################################################################################################
				#GET THE PRIVACY LOSS UNDER TWO ADJACENT DATA SET C	AND 		
			loss = numerator_privacy_loss_one_pair(datasize,epsilon,delta,prior,C,C_adj)
			if loss_max_pair[1] < loss[-1][1]:
				loss_max_pair = loss[-1]
		print loss_max_pair
		loss_in_numerator.append(loss_max_pair[1])

	plot_privacyloss(datasizes, loss_in_numerator, "in numerator")


def get_adjacent_set(x):
	C_adj = deepcopy(x)
	if C_adj[0] == 0:
		C_adj[0] += 1
		C_adj[1] -= 1
	elif C_adj[0] > 0:
		C_adj[0] -= 1
		C_adj[1] += 1
	return C_adj

def privacy_loss_of_size_n(prior, datasize, epsilon, delta):
	Bayesian_Model = BayesInferwithDirPrior(prior, datasize, epsilon, delta)
	Bayesian_Model._set_candidate_scores()
		
	candidates = Bayesian_Model._candidates
		#########################################################################################################################
		#ALL POSSIBLE DATA SETS UNDER SIZE N
	for i in range(len(candidates)):
		candidates[i]._minus(prior)

	candidates = [c._alphas for c in candidates]

	privacyloss_of_n = 0.0
	pair_of_n = []

	candidates = [[datasize-2,2],[datasize-1,1],[datasize, 0],[0,datasize], [1, datasize - 1], [2, datasize - 2]]

	for C in candidates:
	#########################################################################################################################
	#GET THE ADJACENT DATA SET OF DATA SET C			
		C_adj = get_adjacent_set(C)
			#########################################################################################################################
			#GET THE PRIVACY LOSS UNDER TWO ADJACENT DATA SET C	AND 		
		loss = privacy_loss_one_pair(datasize,epsilon,delta,prior,C,C_adj)
		if privacyloss_of_n < loss[-1][1]:
			privacyloss_of_n = loss[-1][1]
			pair_of_n = (C,C_adj)

	return privacyloss_of_n, pair_of_n

def privacy_loss(datasizes,epsilon,delta,prior):
	epsilons = []
	#########################################################################################################################
	#MAX PRIVACY LOSS UNDER SIZE N
	#########################################################################################################################			
	for n in datasizes:
		privacyloss_of_n, pair_of_n = privacy_loss_of_size_n(prior, n, epsilon, delta)

		print " Practical Privacy Loss:", str(privacyloss_of_n), str(pair_of_n)
		epsilons.append(privacyloss_of_n)

	#########################################################################################################################
	#PLOT THE PRIVACY LOSS UNDER DIFFERENT DATA SIZE
	#########################################################################################################################			
	plot_privacyloss(datasizes,epsilons, "")





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
	datasizes = gen_datasizes((2,20),2) + gen_datasizes((25,100),10)# + gen_datasizes((100,500), 100) + gen_datasizes((1000,5000),1000)  + gen_datasizes((10000,50000),10000)
	# datasizes =  gen_datasizes((15,50),10)
	percentage = [0.5,0.5]
	datasets = gen_datasets(percentage, datasizes)
	priors = [dirichlet([1,1])] + gen_priors([5,20], 5, 2) + gen_priors([40,100], 20, 2) + gen_priors([150,300], 50, 2) + gen_priors([400,400], 50, 2)
	privacy_loss_in_numerator(datasizes, epsilon, delta, prior)

