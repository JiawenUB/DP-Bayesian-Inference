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
from gaussian import gaussian
from dpbayesinfer_Gaussian import Bayesian_Inference_Gaussian
from dpbayesinfer_Gaussian import DP_Bayesian_Inference_Gaussian


def gen_datas(sizes, values):
	return [gen_data(sizes[i], values[i]) for i in range(len(sizes))]

def gen_data(n, value = 0.0):
	if value == 0:
		return [random.random() for i in range(n)]
	else:
		return [value for i in range(n)]



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


def plot_mean_error(x,y_list,xstick,xlabel, labels, title):
	plt.figure(figsize=(12,10))
	i = 0	
	for i in range(len(y_list)):
		plt.plot(x,y_list[i],'^',label=labels[i])

	plt.xticks(x,xstick,rotation=70,fontsize=12)
	plt.title(title,fontsize=20)
	plt.xlabel(xlabel,fontsize=15)	
	plt.ylabel('Average Hellinger Distance ',fontsize=15)
	plt.grid()
	plt.legend(loc="best")
	plt.show()


##################################### THE SAMPLING EXPERIMENTS ####################################################################

def sampling_experiments(prior, datas, known_variance, epsilon, delta, rounds):

	DP_model = DP_Bayesian_Inference_Gaussian(Bayesian_Inference_Gaussian(prior,datas[0], known_variance), epsilon, delta)
	exp_results = []
	lap_results = []
	results = []

	for data in datas:
		DP_model._infer_model._update_model(data)
		DP_model._update_model_setting()
		exp_results_for_one_datasize = []
		lap_results_for_one_datasize = []

		for i in range(rounds):
			r = DP_model._exponential_mechanism()
			exp_results_for_one_datasize.append(r - DP_model._infer_model._posterior)
			r = DP_model._laplace_mechanism()
			lap_results_for_one_datasize.append(r - DP_model._infer_model._posterior)
		exp_results.append(sum(exp_results_for_one_datasize)/rounds)
		lap_results.append(sum(lap_results_for_one_datasize)/rounds)
		results.append(exp_results_for_one_datasize)
		results.append(lap_results_for_one_datasize)

	# plot_mean_error([len(data) for data in datas], [exp_results, lap_results],
	# 	[len(data) for data in datas], "Different Data size", ["Exp Mechanism", "Lap Mech"],
	# 	"")
	plot_error_box(results, "Different Data sizes", [len(data) for data in datas], 
		"", ["Exponential Mechanism", "Laplace Mech"], ['navy', 'red'])


if __name__ == "__main__":

	datasize = 99
	epsilon = 1.0
	delta = 0.00000001
	prior = gaussian(0.0, 1.0)
	known_variance = 1.0
	data = gen_data(datasize, 0.1)
	rounds = 1000
	datas = gen_datas([199, 299, 399, 499, 599, 699],[0.1 for i in range(6)])#[9,19,29,39,49,99,199,299],[0.1 for i in range(8)])


	sampling_experiments(prior, datas, known_variance, epsilon, delta, rounds)











	