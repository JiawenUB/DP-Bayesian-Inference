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
from dpbayesinfer import BayesInferwithDirPrior


def row_discrete_probabilities(sample_size,epsilon,delta,prior,observation):
	Bayesian_Model = BayesInferwithDirPrior(prior, sample_size, epsilon, delta)
	Bayesian_Model._set_observation(observation)

	Bayesian_Model._set_candidate_scores()
	Bayesian_Model._set_LS()
	nomalizer = Bayesian_Model._set_SS()

#############################################################################
#ROUND the scores 
#############################################################################

	for i in Bayesian_Model._candidates:
		Bayesian_Model._candidate_scores[i] = round(Bayesian_Model._candidate_scores[i], 6)

#############################################################################
#SORT the scores and put them into bins
#############################################################################

	sorted_scores = sorted(Bayesian_Model._candidate_scores.items(), key=operator.itemgetter(1))
	steps = [-i for i in sorted(list(set(Bayesian_Model._candidate_scores.values())))]

	Bayesian_Model._SS_probabilities.sort()


#############################################################################
#SUM UP the prob within the same bin
#############################################################################

	i = 0

	candidates_classfied_by_steps = []

	probabilities_exp_by_steps = []
	probabilities_lap_1_by_steps = []
	probabilities_lap_2_by_steps = []

	#############################################################################
	#WRITE data into file
	#############################################################################

	f_exp = open("datas/discrete_prob/data_" + str(observation) +"_exp.txt", "w")
	f_lap_1 = open("datas/discrete_prob/data_" + str(observation) +"_lap_sensitivity2.txt", "w")
	f_lap_2 = open("datas/discrete_prob/data_" + str(observation) +"_lap_sensitivity3.txt", "w")

	f_exp.write("Candidates_of_the_same_steps, Hellinger Distance, Probabilities \n")
	f_lap_1.write("Candidates_of_the_same_steps, Hellinger Distance, Probabilities \n")
	f_lap_2.write("Candidates_of_the_same_steps, Hellinger Distance, Probabilities \n")


	while i < len(sorted_scores):
		j = i
		candidates_for_print = []
		candidates_for_classify = []
		while True:
			if (i+1) > len(sorted_scores) or sorted_scores[j][1] != sorted_scores[i][1]:
				break
			candidates_for_print.append(sorted_scores[i][0]._alphas)
			candidates_for_classify.append(sorted_scores[i][0])
			# print sorted_scores[i]
			i += 1
		candidates_classfied_by_steps.append(candidates_for_classify)
		probabilities_exp_by_steps.append(Bayesian_Model._SS_probabilities[j]*(i - j))
		f_exp.write(str(candidates_for_print) + "&" + str(-sorted_scores[j][1]) + "&" + str(Bayesian_Model._SS_probabilities[j]*(i - j)) + "\n")
	f_exp.close()

#############################################################################
#CALCULATE the Laplace prob within the same bin
#############################################################################

	# y = numpy.arange(0,4,1)
	laplace_probabilities_1 = {}
	laplace_probabilities_2 = {}
	p1 = 0.0
	p2 = 0.0
	for i in range(len(Bayesian_Model._candidates)-1):
		r = Bayesian_Model._candidates[i]
		t1 = 1.0
		t2 = 1.0
		# ylist = []
		for j in range(len(r._alphas) - 1):
			a = r._alphas[j] - Bayesian_Model._posterior._alphas[j]
			t1 = t1 * 0.5 * (math.exp(- ((abs(a)) if (a >= 0) else (abs(a) - 1)) / (2.0/epsilon)) - math.exp(- ((abs(a) + 1) if (a >= 0) else (abs(a))) / (2.0/epsilon)))
			t2 = t2 * 0.5 * (math.exp(- ((abs(a)) if (a >= 0) else (abs(a) - 1)) / (3.0/epsilon)) - math.exp(- ((abs(a) + 1) if (a >= 0) else (abs(a))) / (3.0/epsilon)))
		p1 += t1
		p2 += t2

		laplace_probabilities_1[r] = t1 #/ (math.gamma(len(yset)) * (2 ** (len(list(filter(lambda a: a != 0, ylist))))))
		laplace_probabilities_2[r] = t2 #/ (math.gamma(len(yset)) * (2 ** (len(list(filter(lambda a: a != 0, ylist))))))
	#############################################################################
	#SUM UP the tail Laplace prob
	#############################################################################

	laplace_probabilities_1[Bayesian_Model._candidates[-1]] = 1 - p1 
	laplace_probabilities_2[Bayesian_Model._candidates[-1]] = 1 - p2

	for class_i in candidates_classfied_by_steps:
		pro_i_1 = 0.0
		pro_i_2 = 0.0
		candidates_for_print = []
		for c in class_i:
			#print laplace_probabilities[c]
			pro_i_1 += laplace_probabilities_1[c]
			pro_i_2 += laplace_probabilities_2[c]
			candidates_for_print.append(c._alphas)
		probabilities_lap_1_by_steps.append(pro_i_1)
		probabilities_lap_2_by_steps.append(pro_i_2)
		f_lap_1.write(str(candidates_for_print) + "&" + str(-Bayesian_Model._candidate_scores[class_i[0]]) +"&" + str(pro_i_1) + "\n")
		f_lap_2.write(str(candidates_for_print) + "&" + str(-Bayesian_Model._candidate_scores[class_i[0]]) +"&" + str(pro_i_2) + "\n")
	f_lap_1.close()
	f_lap_2.close()	

#############################################################################
#PLOT the prob within the same bin
#############################################################################

	plt.figure()
	plt.plot(steps[-100:], probabilities_exp_by_steps[-100:], 'b', label=(r'$\mathcal{M}^{B}_{\mathcal{H}}$'))
	# plt.plot(T, approximate_bounds, 'g^', label=('Expmech_SS zApproximate Bound'))
	plt.plot(steps[-100:], probabilities_lap_1_by_steps[-100:], 'r', label=('LapMech (sensitivity = 1)'))

	plt.plot(steps[-100:], probabilities_lap_2_by_steps[-100:], 'g', label=('LapMech (sensitivity = 2)'))

	plt.xlabel("c / Hellinger distance from true posterior")
	plt.ylabel("Pr[H(BI(x),r) = c]")
	plt.title("Discrete Probabilities")
	plt.legend()
	plt.grid()
	# plt.savefig("data_"+ str(observation) + "_prior_"+ str(Bayesian_Model._prior._alphas) + "_epsilon_"+ str(epsilon) + "_line.png")
	plt.show()



def discrete_probabilities_from_file(filenames,labels,savename):
	
#############################################################################
#READ the prob from file
#############################################################################

	probabilities_by_steps = []
	steps = []
	for file in filenames:
		f = open(file, "r")
		f.readline()
		s = []
		prob = []
		for line in f:
			l = line.strip("\n").split("&")
			prob.append(float(l[-1]))
			s.append(float(l[-2]))
		probabilities_by_steps.append(prob)
		steps = s

#############################################################################
#PLOT the prob within the same bin
#############################################################################


	plt.figure()
	colors = ["b","r","g"]

	for i in range(len(filenames)):
		plt.plot(steps[-100:], probabilities_by_steps[i][-100:], colors[i], label=(labels[i]))
		# plt.plot(T, approximate_bounds, 'g^', label=('Expmech_SS zApproximate Bound'))
	plt.xlabel("c / (steps from correct answer, in form of Hellinger Distance)")
	plt.ylabel("Pr[H(BI(x),r) = c]")
	plt.title("Discrete Probabilities")
	plt.legend()
	plt.grid()
	plt.show()

def gen_dataset(v, n):
	return [int(n * i) for i in v]

def gen_datasets(v, n_list):
	return [gen_dataset(v,n) for n in n_list]

# def gen_datasets(v, n):
# 	datasizes = 
# 	return [gen_dataset(v,n) for n in datasizes]

def gen_datasizes(r, step):
	return [i*step for i in range(r[0]/step,r[1]/step + 1)]

def gen_priors(r, step, d):
	return [dirichlet([step*i for j in range(d)]) for i in range(r[0]/step,r[1]/step + 1)]


if __name__ == "__main__":

	datasize = 3
	epsilon = 1.0
	delta = 0.00000001
	prior = dirichlet([1,1,1])
	dataset = [1,1,1]
	datasizes = gen_datasizes((600,600),50)#[300] #[8,12,18,24,30,36,42,44,46,48]#,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80]
	percentage = [0.5,0.5]
	datasets = gen_datasets(percentage, datasizes)
	priors = [dirichlet([1,1])] + gen_priors([5,20], 5, 2) + gen_priors([40,100], 20, 2) + gen_priors([150,300], 50, 2) + gen_priors([400,400], 50, 2)
	# for i in range(len(datasizes)):
	# 	row_discrete_probabilities(datasizes[i],epsilon,delta,prior,datasets[i])

	row_discrete_probabilities(datasize,epsilon,delta,prior,dataset)
	# discrete_probabilities_from_file(
	# 	["datas/discrete_prob/data_[1, 1, 1]_exp.txt",
	# 	"datas/discrete_prob/data_[1, 1, 1]_lap_sensitivity2.txt", 
	# 	"datas/discrete_prob/data_[1, 1, 1]_lap_sensitivity3.txt"],
	# 	[r'$\mathcal{M}^{B}_{\mathcal{H}}$',
	# 	"LapMech (sensitivity = 2)",
	# 	"LapMech (sensitivity = 3)"],
	# 	"poster_5")


	

