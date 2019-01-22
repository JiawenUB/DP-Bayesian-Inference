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

#################################################################################
###READ DATA FROM FILE###########################################################
#################################################################################
def read_data(filename, col):
	data = []
	with i = open(filename):
		for line in i:
			t = i.split(" ")
			data.append(t[col])
	return data

def read_datas(folder):
	datas = []
	for f in glob.glob(folder):
		datas.append(read_data(f, 0))
	return datas

def data_process(datas):
	observations = []
	datasizes = []
	for data in datas:
		observations.append([sum(data), len(data) - sum(data)])
		datasizes.append(len(data))
	return datasizes,observations


def run_experiments(times, datasizes, observations,epsilon, delta, prior):
	data = []
	errors = [[],[],[],[],[]]
	for i in range(len(datasizes)):
		observation = observations[i]
		Bayesian_Model = BayesInferwithDirPrior(prior, sum(observation), epsilon, delta)
		Bayesian_Model._set_observation(observation)
		print("start" + str(observation))
		Bayesian_Model._experiments(times)
		print("finished" + str(observation))

		for i in range(len(mean_error)):
			errors[i].append(Bayesian_Model._accuracy[Bayesian_Model._keys[i]])

	plot_error_box(errors, "Data Set", 
		["bike", "cryotherapy", "immunotherapy", "badges"], 
		"Experiments on Real Data", 
		[
		r'Alg 5 - $\mathsf{EHDS}$ ',
		r"Alg 4 - $\mathsf{EHDL}$",
		r"Alg 3 - $\mathsf{EHD}$",
		r'Alg 1 - $\mathsf{LSDim}$ (sensitivity = '+ str(sensitivity2) +')', 
		r'Alg 2 - $\mathsf{LSHist}$ (sensitivity = '+ str(sensitivity1) +')'],
		["skyblue", "navy", "lightblue", "lightcyan", "blueviolet"] )

	return 

#############################################################################
#PLOT THE SAMPLING RESULTS BY 4-QUANTILE BOX PLOTS
#############################################################################

def plot_error_box(data, xlabel, xstick, title, legends, colors):
	l = len(legends)
	plt.figure(figsize=(0.5*len(data),9))
	medianprops = dict(linestyle='--', linewidth=3.0, color='lightblue')
	# meanlineprops = dict(linestyle='--', linewidth=2.5, color='purple')
	bplot = plt.boxplot(data, notch=1, widths=0.4, sym='+', showfliers=False, vert=2, whis=1.2, patch_artist=True, medianprops=medianprops)#, meanprops=meanlineprops, meanline=True,showmeans=True)
	plt.xlabel(xlabel,fontsize=15)
	plt.ylabel('Hellinger Distance',fontsize=15)
	#ax.set_xlim(0.5, len(errors) + 0.5)


	plt.xticks([i*l + (l+1)/2.0 for i in range(len(xstick))],xstick,rotation=40,fontsize=12)
	plt.title(title,fontsize=15)

	for i in range(1, len(bplot["boxes"])/l + 1):
		for j in range(l):
			box = bplot["boxes"][l * (i - 1) + j]
			box.set(color=colors[j], linewidth=1.5)
			box.set(facecolor=colors[j] )
	plt.legend([bplot["boxes"][i] for i in range(l)], legends, loc='best')
	plt.grid()
	plt.show()


#############################################################################
#PLOT THE MEAN SAMPLING RESULTS IN SCATTER 
#############################################################################


def plot_mean_error(x,y_list,xstick,xlabel, ylabel, title):
	plt.figure(figsize=(11,8))
	i = 0	
	for i in range(len(y_list)):
		plt.plot(x, y_list[i],'^',label=ylabel[i])

	plt.xticks(x,xstick,rotation=70,fontsize=12)
	plt.title(title,fontsize=20)
	plt.xlabel(xlabel,fontsize=15)	
	plt.ylabel('Average Hellinger Distance ',fontsize=15)
	plt.grid()
	plt.legend()
	plt.show()




	
#############################################################################
#GENERATING DATA SIZE AND CONRRESPONDING PARAMETER
#############################################################################

def gen_dataset(v, n):
	return [int(n * i) for i in v]

def gen_datasets(v, n_list):
	return [gen_dataset(v,n) for n in n_list]

def gen_datasizes(r, step):
	return [(r[0] + i*step) for i in range(0,(r[1] - r[0])/step + 1)]

def gen_priors(r, step, d):
	return [dirichlet([step*i for j in range(d)]) for i in range(r[0]/step,r[1]/step + 1)]


if __name__ == "__main__":

#############################################################################
#SETTING UP THE PARAMETERS
#############################################################################

	epsilon = 1,0
	delta = 0.00000001
	prior = dirichlet([1,1])


	epsilons = numpy.arange(5, 2, 0.1)
	datasizes, observations = data_process(read_datas("/data/*.txt"))
	run_experiments(1000, datasizes, observations,epsilon, delta, prior)


