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


def ratio_of_adj_local_sensitivity(datasizes, prior, beta, delta):
	ratios = []
	for datasize in datasizes:
		ratio_max = 0.0
		max_position = 0
		candidates = [[datasize-2,2],[datasize-1,1],[datasize, 0],[0,datasize], [1, datasize - 1], [2, datasize - 2]]
		candidates = [list(i) for i in numpy.array(candidates) +1]
		for C in candidates:
		#########################################################################################################################
		#GET THE ADJACENT DATA SET OF DATA SET C			
			C_adj = get_adjacent_set(C)
				#########################################################################################################################
				#GET THE PRIVACY LOSS UNDER TWO ADJACENT DATA SET C	AND 		
			ratio = (1.0 / dirichlet(C_adj)._hellinger_sensitivity() - 1.0 / dirichlet(C)._hellinger_sensitivity())
			if ratio_max < ratio:
				ratio_max = ratio
				max_position = C

		ratios.append(ratio_max)
		print ratio_max, max_position
	plot_ratio_of_LS(ratios, datasizes)

def sensitivities(datasets, prior, beta, delta, label):
	sensitivities = []
	for dataset in datasets:
		Bayesian_Model = BayesInferwithDirPrior(prior, sum(dataset), epsilon, delta)
		Bayesian_Model._set_observation(dataset)
		Bayesian_Model._set_candidate_scores()
		Bayesian_Model._set_local_sensitivities()
		candidates = Bayesian_Model._candidates
		for c in candidates:
		#########################################################################################################################
		#GET THE ADJACENT DATA SET OF DATA SET C			
			if (label == r"Local Sensitivity ($LS(x')$)"):
				sensitivities.append(Bayesian_Model._LS_Candidates[c])
			elif (label == r"Our Sensitivity ($\frac{1}{\frac{1}{LS(x')} +\gamma \cdot d(x,x')}$)"):
				sensitivities.append((1.0 / (1.0/Bayesian_Model._LS_Candidates[c] + 1.0 * Hamming_Distance(Bayesian_Model._observation_counts, [c._alphas[i] - Bayesian_Model._prior._alphas[i] for i in range(Bayesian_Model._prior._size)]))))
	return sensitivities, [list(numpy.array(c._alphas) - 1) for c in candidates]

def sensitivities_2(datasizes, prior, beta, delta, label, gamma=1.0):
	sensitivities = []
	candidates = []
	for datasize in datasizes:
		Bayesian_Model = BayesInferwithDirPrior(prior, datasize, epsilon, delta, gamma)
		Bayesian_Model._set_observation(gen_dataset([0.5,0.5],datasize))
		Bayesian_Model._set_candidate_scores()
		Bayesian_Model._set_local_sensitivities()
		candidates = [list(numpy.array(c._alphas) - 1) for c in Bayesian_Model._candidates]
		if(label == r"Local Sensitivity ($LS(x)$)"):
			for c in Bayesian_Model._candidates:
				sensitivities.append(Bayesian_Model._LS_Candidates[c])
		elif(label == r"Our Sensitivity ($\max_{x'}(\frac{1}{\frac{1}{LS(x')} +\gamma \cdot d(x,x')})$)"):		
			for c in candidates:
				Bayesian_Model._set_observation(c)
				Bayesian_Model._set_up_exp_mech_with_LS()
				Bayesian_Model._set_up_exp_mech_with_gamma_SS()
				sensitivities.append(Bayesian_Model._gamma_SS)
	return sensitivities, candidates

def gen_sensitivities(datasizes, prior, beta, delta, labels, gammas):
	y = []
	x = []
	legends = []
	for label in labels:
		if(label == r"Our Sensitivity ($\max_{x'}(\frac{1}{\frac{1}{LS(x')} +\gamma \cdot d(x,x')})$)"):		
			for gamma in gammas:
				s,x = sensitivities_2(datasizes, prior, beta, delta, label, gamma)
				y.append(s)
				legends.append(label + r"$\gamma = $" + str(gamma))
		else:
			s,x = sensitivities_2(datasizes, prior, beta, delta, label)
			y.append(s)
			legends.append(label)

	plot_sensitivities(y,x,legends, str(datasizes))


def get_adjacent_set(x):
	C_adj = deepcopy(x)
	if C_adj[0] == 0:
		C_adj[0] += 1
		C_adj[1] -= 1
	elif C_adj[0] > 0:
		C_adj[0] -= 1
		C_adj[1] += 1
	return C_adj

def plot_ratio_of_LS(x, y):
	plt.figure()
	plt.title(("Ratio of Local Sensitivity for Adjacent Data"))
	plt.plot(x,y, 'bo-', label=(r'$\max(\frac{1}{LS(x\')} - \frac{1}{LS(x)})$'))
	plt.xlabel("Data Size")
	plt.ylabel(r"$\max(\frac{1}{LS(x\')} - \frac{1}{LS(x)})$ (Hellinger Distance)")
	plt.grid()
	plt.legend(loc='best')
	plt.show()

def plot_sensitivities(y,xstick,labels, title):
	plt.figure()
	x = range(len(xstick))
	plt.plot(x,y[0], 'bo', label=labels[0])
	for i in range(1,len(y)):
		plt.plot(x,y[i],label=labels[i])
	plt.xlabel(r"Different Observed Data Sets ($x$)")
	plt.xticks(x,xstick,rotation=70,fontsize=6)
	plt.ylabel("Sensitivity")
	plt.title("Sensitivities for Data sets of Size " + title)
	plt.legend(loc='best',fontsize=6)
	plt.grid()
	plt.show()

def gen_dataset(v, n):
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

	datasize = 20
	epsilon = 1.0
	delta = 0.00000001
	prior = dirichlet([1,1])
	dataset = [50,50]
	epsilons = numpy.arange(0.1, 2, 0.1)
	datasizes = gen_datasizes((100,100),2)# + gen_datasizes((25,100),10) + gen_datasizes((100,500), 100) + gen_datasizes((1000,5000),1000)#  + gen_datasizes((10000,50000),10000)
	percentage = [0.5,0.5]
	datasets = gen_datasets(percentage, datasizes)
	priors = [dirichlet([1,1])] + gen_priors([5,20], 5, 2) + gen_priors([40,100], 20, 2) + gen_priors([150,300], 50, 2) + gen_priors([400,400], 50, 2)
	gammas = gen_gammas([0,11], 2, 0.1) + gen_gammas([15,30], 5, 0.1)
	print gammas

	gen_sensitivities(datasizes, prior, beta, delta, [r"Local Sensitivity ($LS(x)$)",
		r"Our Sensitivity ($\max_{x'}(\frac{1}{\frac{1}{LS(x')} +\gamma \cdot d(x,x')})$)"], gammas)
