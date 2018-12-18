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
	datasizes = gen_datasizes((2,20),2) + gen_datasizes((25,100),10) + gen_datasizes((100,500), 100) + gen_datasizes((1000,5000),1000)#  + gen_datasizes((10000,50000),10000)
	# datasizes =  gen_datasizes((15,50),10)
	percentage = [0.5,0.5]
	datasets = gen_datasets(percentage, datasizes)
	priors = [dirichlet([1,1])] + gen_priors([5,20], 5, 2) + gen_priors([40,100], 20, 2) + gen_priors([150,300], 50, 2) + gen_priors([400,400], 50, 2)

	ratio_of_adj_local_sensitivity(datasizes, prior, beta, delta)

