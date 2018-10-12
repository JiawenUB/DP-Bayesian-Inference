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


def LAPLACE_CDF(interval, scale):
	if interval[0] >= 0.0:
		return (1 - 0.5 * math.exp( (-interval[1]*1.0/scale))) - (1 - 0.5 * math.exp( (-interval[0]/scale)))
	else:
		return (0.5 * math.exp( (interval[1]*1.0/scale))) - (0.5 * math.exp( (interval[0]/scale)))

#############################################################################
#CALCULATING THE DISCRETE PROBABILITIES
#############################################################################

def row_discrete_probabilities(sample_size,epsilon,delta,prior,observation):

	Bayesian_Model = BayesInferwithDirPrior(prior, sample_size, epsilon, delta)
	Bayesian_Model._set_observation(observation)

	Bayesian_Model._set_candidate_scores()
	Bayesian_Model._set_LS()
	Bayesian_Model._set_SS()


#############################################################################
#SPLIT THE BINS
#############################################################################
	
	Candidate_bins_by_step = {}
	for r in Bayesian_Model._candidates:
		if str(sorted(r._alphas)) not in Candidate_bins_by_step.keys():
			Candidate_bins_by_step[str(sorted(r._alphas))] = []
			for c in Bayesian_Model._candidates:
				if set(c._alphas) == set(r._alphas):
					Candidate_bins_by_step[str(sorted(r._alphas))].append(c)
			


#############################################################################
#SUM UP the prob within the same bin
#############################################################################

	i = 0


	#############################################################################
	#WRITE data into file
	#############################################################################

	f_exp = open("datas/discrete_prob/data_" + str(observation) +"_exp.txt", "w")

	f_exp.write("Candidates_of_the_same_steps&Hellinger Distance&Probabilities \n")


	probability_distance_pairs_in_exp =[]

	nomalizer = 0.0

	#############################################################################
	#CALCULATING THE UNOMARLIZED PROBABILITY OF EACH BIN
	#############################################################################

	for key,item in Candidate_bins_by_step.items():
		# print key, len(item)
		#THE HELLINGER DISTANCE OF THIS BIN
		distance = -Bayesian_Model._candidate_scores[item[0]]
		
		#THE PROBABILITY OF THIS BIN
		prob = (len(item) * math.exp(epsilon * Bayesian_Model._candidate_scores[item[0]]/(2 * Bayesian_Model._SS)))

		probability_distance_pairs_in_exp.append((distance, prob, [i._alphas for i in item]))

		nomalizer += prob

	#############################################################################
	#NOMARLIZING PROBABILITY
	#############################################################################

	probability_distance_pairs_in_exp = [(t[0],t[1]/nomalizer,t[2]) for t in probability_distance_pairs_in_exp]

#############################################################################
#SORT AND SPLIT THE PROBABILITY FROM PAIRSTHE #WRITE DATAS INTO FILE
#############################################################################
	probability_distance_pairs_in_exp.sort()
	# print probability_distance_pairs_in_exp

	for triple in probability_distance_pairs_in_exp:
		#WRITE DATAS INTO FILE
		f_exp.write(str(triple[2]) + "&" + str(triple[0]) + "&" + str(triple[1]/nomalizer) + "\n")
	
	f_exp.close()

	t, exp, _ = zip(*probability_distance_pairs_in_exp)



#############################################################################
#CALCULATE the Laplace prob within the same bin
#############################################################################
	
	#############################################################################
	#SENSITIVITY SETTING
	#############################################################################
	
	f_lap_1 = open("datas/discrete_prob/data_" + str(observation) +"_lap_sensitivity2.txt", "w")
	f_lap_2 = open("datas/discrete_prob/data_" + str(observation) +"_lap_sensitivity3.txt", "w")
	f_lap_1.write("Candidates_of_the_same_steps, Hellinger Distance, Probabilities \n")
	f_lap_2.write("Candidates_of_the_same_steps, Hellinger Distance, Probabilities \n")

	def sensitivity_setting(dimension):
		if dimension == 2:
			return (1.0,2.0)
		else:
			return (2.0,dimension*1.0)

	sensitivity = sensitivity_setting(len(prior._alphas))
	
	#############################################################################
	#CALCULATING THE LAPLACE PROB
	#############################################################################
	
	probability_distance_pairs_in_lap_1 = []
	probability_distance_pairs_in_lap_2 = []
	for key,item in Candidate_bins_by_step.items():
		p1 = 0.0
		p2 = 0.0
		for r in item:
			t1 = 1.0
			t2 = 1.0
			#THE LAPLACE PROBABILITY OF THIS BIN
			for j in range(len(r._alphas) - 1):
				a = r._alphas[j] - Bayesian_Model._posterior._alphas[j]
				t1 = t1 * LAPLACE_CDF((a,a+1), sensitivity[0]/epsilon)
				t2 = t2 * LAPLACE_CDF((a,a+1), sensitivity[1]/epsilon)
			p1 += t1
			p2 += t2
		probability_distance_pairs_in_lap_1.append((-Bayesian_Model._candidate_scores[item[0]],p1,[i._alphas for i in item]))
		probability_distance_pairs_in_lap_2.append((-Bayesian_Model._candidate_scores[item[0]],p2,[i._alphas for i in item]))
	


#############################################################################
#SORT AND SPLIT THE PROBABILITY FROM BINS WRITEINTO FILE
#############################################################################
	probability_distance_pairs_in_lap_1.sort()
	probability_distance_pairs_in_lap_2.sort()

	for i in range(len(probability_distance_pairs_in_lap_1)):
		e1 = probability_distance_pairs_in_lap_1[i]
		e2 = probability_distance_pairs_in_lap_2[i]
		f_lap_1.write(str(e1[2]) + "&" + str(e1[0]) +"&" + str(e1[1]) + "\n")
		f_lap_2.write(str(e2[2]) + "&" + str(e2[0]) +"&" + str(e2[1]) + "\n")

	f_lap_1.close()
	f_lap_2.close()	

	t, lap1, _ = zip(*(probability_distance_pairs_in_lap_1))
	t, lap2, _ = zip(*(probability_distance_pairs_in_lap_2))

#############################################################################
#TAIL PROBABILITY
#############################################################################
	lap1 = list(lap1)
	lap2 = list(lap2)

	lap1[-1] = 1.0 - sum(lap1[:-1])
	lap2[-1] = 1.0 - sum(lap2[:-1])

#############################################################################
#PLOT the prob within the same bin
#############################################################################

	#############################################################################
	#LABELS SETTING
	#############################################################################

	def label_setting():
		return [r'$\mathcal{M}^{B}_{\mathcal{H}}$', 'LapMech (sensitivity = ' + str(sensitivity[0]) + ')', 'LapMech (sensitivity = ' + str(sensitivity[1]) + ')']

	labels = label_setting()

	#############################################################################
	#STEP SETTING
	#############################################################################

	step = t


	#############################################################################
	#PLOTTING
	#############################################################################

	plt.figure()
	plt.plot(step, exp, 'b', label=(labels[0]))

	plt.plot(step, lap1, 'r', label=(labels[1]))

	plt.plot(step, lap2, 'g', label=(labels[2]))

	#############################################################################
	#PLOT FEATURE SETTING
	#############################################################################

	plt.xlabel("c / Hellinger distance from true posterior")
	plt.ylabel("Pr[H(BI(x),r) = c]")
	plt.title("Discrete Probabilities", fontsize=15)
	plt.legend()
	plt.grid()
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


#############################################################################
#GENERATING DATA SIZE AND CONRRESPONDING PARAMETER
#############################################################################

def gen_dataset(v, n):
	return [int(n * i) for i in v]

def gen_datasets(v, n_list):
	return [gen_dataset(v,n) for n in n_list]

def gen_datasizes(r, step):
	return [i*step for i in range(r[0]/step,r[1]/step + 1)]

def gen_priors(r, step, d):
	return [dirichlet([step*i for j in range(d)]) for i in range(r[0]/step,r[1]/step + 1)]


if __name__ == "__main__":

#############################################################################
#SETTING UP THE PARAMETERS
#############################################################################
	datasize = 300
	epsilon = 1.0
	delta = 0.00000001
	prior = dirichlet([1,1,1])
	dataset = [100,100,100]

#############################################################################
#SETTING UP THE PARAMETERS WHEN DOING GROUPS EXPERIMENTS
#############################################################################
	
	datasizes = gen_datasizes((600,600),50)
	percentage = [0.5,0.5]
	datasets = gen_datasets(percentage, datasizes)
	priors = [dirichlet([1,1])] + gen_priors([5,20], 5, 2) + gen_priors([40,100], 20, 2) + gen_priors([150,300], 50, 2) + gen_priors([400,400], 50, 2)

#############################################################################
#DO PLOTS BY COMPUTING THE PROBABILITIES FOR GROUP EXPERIMENTS
#############################################################################

	# for i in range(len(datasizes)):
	# 	row_discrete_probabilities(datasizes[i],epsilon,delta,prior,datasets[i])

#############################################################################
#DO PLOTS BY COMPUTING THE PROBABILITIES
#############################################################################

	row_discrete_probabilities(datasize,epsilon,delta,prior,dataset)

#############################################################################
#DO PLOTS BY READING THE PROB FROM FILES
#############################################################################

	# discrete_probabilities_from_file(
	# 	["datas/discrete_prob/data_[1, 1, 1]_exp.txt",
	# 	"datas/discrete_prob/data_[1, 1, 1]_lap_sensitivity2.txt", 
	# 	"datas/discrete_prob/data_[1, 1, 1]_lap_sensitivity3.txt"],
	# 	[r'$\mathcal{M}^{B}_{\mathcal{H}}$',
	# 	"LapMech (sensitivity = 2)",
	# 	"LapMech (sensitivity = 3)"],
	# 	"poster_5")


	

