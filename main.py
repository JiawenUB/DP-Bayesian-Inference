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


def draw_error(errors, model, filename):
	# plt.subplots(nrows=len(errors), ncols=1, figsize=(18, len(errors) * 5.0))
	# plt.tight_layout(pad=2, h_pad=4, w_pad=2, rect=None)
	rows = 1
	data = []
	title = []
	for key,item in errors.items():
		data.append(item)
		title.append(key)

	fig, ax = plt.subplots()
	bplot = plt.boxplot(data, notch=1, widths=0.4, sym='+', vert=2, whis=1.5,patch_artist=True)
	plt.xlabel("different mechanisms")
	plt.ylabel('Distance Based on Hellinger Distance')
	#ax.set_xlim(0.5, len(errors) + 0.5)

	plt.xticks(range(1, len(errors)+1),title)
	plt.title('Accuracy / data set:' + str(model._observation_counts) + ", posterior: " + str(model._posterior._alphas) + ", epsilon:" + str(model._epsilon))
	for box in bplot["boxes"]:
		box.set(color='navy', linewidth=1.5)
		box.set(facecolor='lightblue')
	plt.show()
	# plt.savefig(filename)
	return 

def draw_error_l1(errors, model, filename):
	# plt.subplots(nrows=len(errors), ncols=1, figsize=(18, len(errors) * 5.0))
	# plt.tight_layout(pad=2, h_pad=4, w_pad=2, rect=None)
	rows = 1
	data = []
	title = []
	for key,item in errors.items():
		data.append(item)
		title.append(key)
		# plt.subplot(len(errors), 1, rows)
	fig, ax = plt.subplots()
	bplot = plt.boxplot(data, notch=1, widths=0.4, sym='+', vert=2, whis=1.5,patch_artist=True)
	plt.xlabel("different mechanisms")
	plt.ylabel('Distance Based on L1 Norm Distance')
	#ax.set_xlim(0.5, len(errors) + 0.5)

	plt.xticks(range(1, len(errors)+1),title)
	plt.title('Accuracy / data set:' + str(model._observation_counts) + ", posterior: " + str(model._posterior._alphas) + ", epsilon:" + str(model._epsilon))
	for box in bplot["boxes"]:
		box.set(color='navy', linewidth=1.5)
		box.set(facecolor='lightblue' )
	plt.show()
	# plt.savefig(filename)
	return




def accuracy_study_discrete(sample_size,epsilon,delta,prior,observation):
	Bayesian_Model = BayesInferwithDirPrior(prior, sample_size, epsilon, delta)
	Bayesian_Model._set_observation(observation)

	Bayesian_Model._set_candidate_scores()
	Bayesian_Model._set_LS()
	nomalizer = Bayesian_Model._set_SS()

	sorted_scores = sorted(Bayesian_Model._candidate_scores.items(), key=operator.itemgetter(1))
	steps = [-i for i in sorted(list(set(Bayesian_Model._candidate_scores.values())))]

	Bayesian_Model._SS_probabilities.sort()
	# print Bayesian_Model._SS_probabilities

	i = 0
	# print sorted_scores

	candidates_classfied_by_steps = []

	probabilities_exp_by_steps = []
	probabilities_lap_by_steps = []

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
		# print "ExpMech: Pr[H(BI(x), r) = " + str(-sorted_scores[j][1]) + " ] = " + str(Bayesian_Model._SS_probabilities[j]*(i - j)) + " (r = " + str(candidates_for_print) +")"
   
	# y = numpy.arange(0,4,1)
	laplace_probabilities = {}
	for i in range(len(Bayesian_Model._candidates)):
		r = Bayesian_Model._candidates[i]
		t = 1.0
		# ylist = []
		for j in range(len(r._alphas) - 1):
			a = r._alphas[j] - Bayesian_Model._posterior._alphas[j]
			t = t * 0.5 * (math.exp(- ((abs(a)) if (a >= 0) else (abs(a) - 1)) / (2.0/epsilon)) - math.exp(- ((abs(a) + 1) if (a >= 0) else (abs(a))) / (2.0/epsilon)))
		# 	ylist.append(a)
		# yset = set(ylist)
		laplace_probabilities[r] = t #/ (math.gamma(len(yset)) * (2 ** (len(list(filter(lambda a: a != 0, ylist))))))

	for class_i in candidates_classfied_by_steps:
		pro_i = 0.0
		candidates_for_print = []
		for c in class_i:
			#print laplace_probabilities[c]
			pro_i += laplace_probabilities[c]
			candidates_for_print.append(c._alphas)
		probabilities_lap_by_steps.append(pro_i)
		
		# print "Laplace: Pr[H(BI(x), r) = " + str(-Bayesian_Model._candidate_scores[class_i[0]]) + " ] = " + str(pro_i) + " (r = " + str(candidates_for_print) +")"
	print "ExpMech: Pr[H(BI(x), r) = 0.0 ] = " + str(probabilities_exp_by_steps[-1])
	print "Laplace: Pr[H(BI(x), r) = 0.0 ] = " + str(probabilities_lap_by_steps[-1])



	plt.plot(steps[-100:], probabilities_exp_by_steps[-100:], 'ro', label=('Exp Mech'))
	# plt.plot(T, approximate_bounds, 'g^', label=('Expmech_SS zApproximate Bound'))
	plt.plot(steps[-100:], probabilities_lap_by_steps[-100:], 'b^', label=('Laplace Mech'))
	plt.xlabel("c / (steps from correct answer, in form of Hellinger Distance)")
	plt.ylabel("Pr[H(BI(x),r) = c]")
	plt.title("datasize: "+ str(sample_size) + ", x: "+ str(observation) + ", BI(x): beta"+ str(Bayesian_Model._posterior._alphas) + ", epsilon: "+ str(epsilon))
	plt.legend()
	plt.grid()
	plt.show()



def global_epsilon_study(sample_sizes,epsilon,delta,prior):
	epsilons = []
	for n in sample_sizes:
		Bayesian_Model = BayesInferwithDirPrior(prior, n, epsilon, delta)
		Bayesian_Model._set_candidate_scores()
		candidates = Bayesian_Model._candidates
		for i in range(len(candidates)):
			candidates[i]._minus(prior)
		# print candidates
		epsilon_of_n = 0.0
		pair_of_n = []
		for c in candidates:
			temp = deepcopy(c._alphas)
			temp[0] -= 1
			temp[1] += 1
			# print c._alphas, temp
			t = epsilon_study(n,epsilon,delta,prior,c._alphas,temp)
			if epsilon_of_n < t:
				epsilon_of_n = t
				pair_of_n = [c._alphas,temp]

		print " Actually epsilon value:", str(epsilon_of_n), str(pair_of_n)
		epsilons.append(epsilon_of_n)

	plt.figure()
	plt.title(("epsilon study wrt. the data size/ prior: " + str(prior._alphas)))

	plt.plot(sample_sizes,epsilons, 'bo-', label=('Exp Mech'))
	plt.xlabel("Data Size")
	plt.ylabel("maximum epsilon of data size n")
	plt.grid()
	plt.legend(loc='best')
	plt.show()



def epsilon_study(sample_size,epsilon,delta,prior,x1, x2):
	x1_probabilities = epsilon_study_discrete_probabilities(sample_size,epsilon,delta,prior,x1)
	x2_probabilities = epsilon_study_discrete_probabilities(sample_size,epsilon,delta,prior,x2)
	# print x1_probabilities
	accuracy_epsilons = {}
	for key, item in x1_probabilities.items():
		for k,i in x2_probabilities.items():
			if key._alphas == k._alphas:
				accuracy_epsilons[str(key._alphas)] = math.log(item / i)

	sorted_epsilons = sorted(accuracy_epsilons.items(), key=operator.itemgetter(1))
	# print sorted_epsilons
	# print x1, x2
	# if sorted_epsilons[-1][1] > abs(sorted_epsilons[0][1]):
	# 	print sorted_epsilons[-1]
	# else:
	# 	print sorted_epsilons[0]
	return max(sorted_epsilons[-1][1], abs(sorted_epsilons[0][1]))
	
	for key,value in sorted_epsilons:
		print "Pr[ ( M(x1) = " + key + ") / ( M(x2) = " + key + ") ] = exp(" + str(value) + ")"

	y = [value for key, value in sorted_epsilons]

	x = range(len(sorted_epsilons))

	xlabel = [key for key, value in sorted_epsilons]
	plt.figure(figsize=(15,8))
	plt.plot(x[0:5], y[0:5], 'bs-', label=('Exp Mech'))
	# plt.plot(T, approximate_bounds, 'g^', label=('Expmech_SS zApproximate Bound'))
	plt.xlabel("z / (candiates)")
	plt.ylabel("Pr[(Pr[ ( M(x1) = z) / ( M(x2) = z) ])] = exp(y)")

	plt.title("datasize: "+ str(sample_size) + ", x1: "+ str(x1) + ", x2: "+ str(x2) + ", epsilon: "+ str(epsilon))
	plt.legend()
	plt.xticks(x,xlabel,rotation=70,fontsize=8)
	plt.grid()
	plt.show()


def epsilon_study_discrete_probabilities(sample_size,epsilon,delta,prior,observation):
	Bayesian_Model = BayesInferwithDirPrior(prior, sample_size, epsilon, delta)
	Bayesian_Model._set_observation(observation)

	Bayesian_Model._set_candidate_scores()
	Bayesian_Model._set_LS()
	Bayesian_Model._set_SS()
	# print Bayesian_Model._SS_probabilities
	# print sorted_scores

	probabilities_exp = {}

	for i in range(len(Bayesian_Model._candidates)):
		z = Bayesian_Model._candidates[i]
		probabilities_exp[z] = Bayesian_Model._SS_probabilities[i]
		# print "Pr[ z = " + str(z._alphas) + " ] = " + str(Bayesian_Model._SS_probabilities[i])
	
	return probabilities_exp

def accuracy_VS_epsilon(sample_size,epsilons,delta,prior,observation):
	exp_accuracy = []
	lap_accuracy = []
	exp_accuracy_average = []
	lap_accuracy_average = []

	for epsilon in epsilons:
		Bayesian_Model = BayesInferwithDirPrior(prior, sample_size, epsilon, delta)
		Bayesian_Model._set_observation(observation)
		Bayesian_Model._experiments(150)
		exp_accuracy.append(Bayesian_Model._accuracy[Bayesian_Model._keys[3]])
		lap_accuracy.append(Bayesian_Model._accuracy[Bayesian_Model._keys[0]])
		exp_accuracy_average.append(statistics.mean(Bayesian_Model._accuracy[Bayesian_Model._keys[3]]))
		lap_accuracy_average.append(statistics.mean(Bayesian_Model._accuracy[Bayesian_Model._keys[0]]))
	plt.figure(figsize=(15,8))

	for i in range(len(epsilons)):
		for a in exp_accuracy[i]:
			plt.plot(epsilons[i], a, c = 'r', marker = 'o', markeredgecolor = 'none', alpha = 0.2)
		for b in lap_accuracy[i]:
			plt.plot(epsilons[i], b, c = 'b', marker = 'o', markeredgecolor = 'none', alpha = 0.2)

	plt.plot(epsilons,exp_accuracy_average, 'ro-', label=('Exp Mech (mean)'))
	plt.plot(epsilons,lap_accuracy_average, 'bo-', label=('Laplace Mech (mean)'))

	plt.title("datasize: "+ str(sample_size) + ", x: "+ str(observation))

	plt.ylabel("Accuracy/Hellinger Distance")
	plt.xlabel("epsilon")

	plt.legend(loc="best")
	plt.grid()
	plt.show()

	return

def hellinger_vs_l1norm(base_distribution):
	l1s = numpy.arange(1, 8, 1)
	
	#l1s = []
	hellingers = []
	xstick = []
	for i in l1s:
		# getcontext().prec = 3
		label = deepcopy(base_distribution._alphas)
		label[0] += i
		label[1] -= i
		# label[0] -= Decimal(i) / Decimal(100.0)
		# label[1] += Decimal(i) / Decimal(100.0)
		# label[0] = float(label[0])
		# label[1] = float(label[1])
		xstick.append(label)
		hellingers.append(base_distribution - dirichlet(label))
		# hellingers.append(Hellinger_Distance_dirichlet(base_distribution, dirichlet(label)))

	plt.figure(figsize=(15,10))
	plt.plot(l1s, hellingers, label="hellinger")
	plt.plot(l1s,0.2344 * l1s,label='linear')
	plt.ylabel("Hellinger Distance")
	plt.xlabel("l1 Norm")
	plt.legend()

	plt.xticks(l1s,xstick, rotation=70,fontsize=8)

	plt.show()


def plot_mean_error(x,y_list,xstick,xlabel,title):
	plt.figure(figsize=(12,10))
	i = 0
	ylabel=['Our ExpMech',"Laplace Mechanism", "Laplace Mech of Zhang"]
	
	for i in range(3):
		plt.plot(x,y_list[i],'^',label=ylabel[i])

	plt.xticks(x,xstick,rotation=70,fontsize=12)
	plt.title(title,fontsize=20)
	plt.xlabel(xlabel,fontsize=15)	
	plt.ylabel('Mean Error / Hellinegr Distance ',fontsize=15)
	plt.grid()
	plt.legend()
	plt.show()

def plot_error_box(data, xlabel,xstick,title):
	plt.figure(figsize=(12,10))
	bplot = plt.boxplot(data, notch=1, widths=0.4, sym='+', vert=2, whis=1.5,patch_artist=True)
	plt.xlabel(xlabel,fontsize=15)
	plt.ylabel('Accuracy / Hellinegr Distance',fontsize=15)
	#ax.set_xlim(0.5, len(errors) + 0.5)

	plt.xticks(range(0, len(data)),xstick,rotation=70,fontsize=12)
	plt.title(title,fontsize=20)
	for i in range(1, len(bplot["boxes"])/2 + 1):
		box = bplot["boxes"][2 * (i - 1)]
		box.set(color='navy', linewidth=1.5)
		box.set(facecolor='lightblue' )
		box = bplot["boxes"][2 * (i - 1) + 1]
		box.set(color='navy', linewidth=1.5)
		box.set(facecolor='darkkhaki' )
	plt.grid()
	plt.show()


def accuracy_VS_datasize(epsilon,delta,prior,observations,datasizes):
	data = []
	xstick = []
	mean_error = [[],[],[]]
	for observation in observations:
		print str(observations)
		Bayesian_Model = BayesInferwithDirPrior(prior, sum(observation), epsilon, delta)
		Bayesian_Model._set_observation(observation)
		Bayesian_Model._experiments(3000)
		mean_error[0].append(Bayesian_Model._accuracy_mean[Bayesian_Model._keys[3]])
		mean_error[1].append(Bayesian_Model._accuracy_mean[Bayesian_Model._keys[0]])
		mean_error[2].append(Bayesian_Model._accuracy_mean[Bayesian_Model._keys[4]])
		# data.append(Bayesian_Model._accuracy[Bayesian_Model._keys[3]])
		# data.append(Bayesian_Model._accuracy[Bayesian_Model._keys[0]])
		# data.append(Bayesian_Model._accuracy[Bayesian_Model._keys[4]])
		# xstick.append(str(observation) + "/ExpMech")
		# xstick.append(str(observation) + "/Laplace")
		# xstick.append(str(observation) + "/Laplace_Zhang")
	print('Accuracy / prior: ' + str(prior._alphas) + ", delta: " + str(delta) + ", epsilon:" + str(epsilon))

	plot_mean_error(range(0,len(observations)),mean_error,datasizes,"Different Data sizes","Mean Error VS. Data Size")
	# plot_error_box(data,"Different Datasizes",xstick,"Accuracy VS. Data Size")
	return




def accuracy_VS_prior(sample_size,epsilon,delta,priors,observation):
	data = []
	xstick = []
	xstick_mean = []
	mean_error = [[],[]]
	for prior in priors:
		Bayesian_Model = BayesInferwithDirPrior(prior, sample_size, epsilon, delta)
		Bayesian_Model._set_observation(observation)
		Bayesian_Model._experiments(500)
		data.append(Bayesian_Model._accuracy[Bayesian_Model._keys[3]])
		data.append(Bayesian_Model._accuracy[Bayesian_Model._keys[0]])
		mean_error[0].append(Bayesian_Model._accuracy_mean[Bayesian_Model._keys[3]])
		mean_error[1].append(Bayesian_Model._accuracy_mean[Bayesian_Model._keys[0]])
		xstick_mean.append(str(prior._alphas))

		xstick.append(str(prior._alphas) + "/ExpMech")
		xstick.append(str(prior._alphas) + "/Laplace")

	print('Accuracy / observation: ' + str(observation) + ", delta: " + str(delta) + ", epsilon:" + str(epsilon))
	
	# plot_mean_error(range(0,len(priors)),mean_error,xstick_mean,"Different Prior Distributions","Mean Accuracy VS. Prior Distribution")
	
	plot_error_box(data,"Different Prior Distributions",xstick,"Accuracy VS. Prior Distribution")

	return

def accuracy_VS_mean(sample_size,epsilon,delta,prior):
	data = []
	xlabel = []
	temp = BayesInferwithDirPrior(prior, sample_size, epsilon, delta)
	temp._set_candidate_scores()
	observations = temp._candidates
	for i in range(len(observations)):
		observations[i]._minus(prior)
	for observation in observations:
		# observation = [int(i * sample_size) for i in mean[:-1]]
		# observation.append(sample_size - sum(observation))
		# print observation
		# print observation._alphas
		Bayesian_Model = BayesInferwithDirPrior(prior, sample_size, epsilon, delta)
		Bayesian_Model._set_observation(observation._alphas)
		Bayesian_Model._experiments(500)
		data.append(Bayesian_Model._accuracy[Bayesian_Model._keys[3]])
		data.append(Bayesian_Model._accuracy[Bayesian_Model._keys[0]])
		xlabel.append(str(observation._alphas) + "/ExpMech")
		xlabel.append(str(observation._alphas) + "/Laplace")

	plt.figure(figsize=(18,10))
	bplot = plt.boxplot(data, notch=1, widths=0.4, sym='+', vert=2, whis=1.5,patch_artist=True)
	plt.xlabel("different observed data sets",fontsize=16)
	plt.ylabel('Accuracy / Hellinegr Distance',fontsize=16)
	#ax.set_xlim(0.5, len(errors) + 0.5)

	plt.xticks(range(1, len(data)+1),xlabel,rotation=70,fontsize=13)
	plt.title("Accuracy VS. Data Variance",fontsize=20)
	print 'Accuracy / data_size: ' + str(sample_size) +  ', prior:' + str(prior._alphas) + ", delta: " + str(delta) + ", epsilon:" + str(epsilon)
	for i in range(1, len(bplot["boxes"])/2 + 1):
		box = bplot["boxes"][2 * (i - 1)]
		box.set(color='navy', linewidth=1.5)
		box.set(facecolor='lightblue' )
		box = bplot["boxes"][2 * (i - 1) + 1]
		box.set(color='navy', linewidth=1.5)
		box.set(facecolor='darkkhaki' )
	plt.grid()
	plt.show()
	return

def accuracy_VS_dimension(sample_sizes, epsilon, delta):
	data = []
	xlabel = []
	for n in sample_sizes:
		for d in range(2,5):
			observation = [n for i in range(d)]
			prior = Dir([1 for i in range(d)])
			Bayesian_Model = BayesInferwithDirPrior(prior, n*d, epsilon, delta)
			Bayesian_Model._set_observation(observation)
			Bayesian_Model._experiments(500)
			data.append(Bayesian_Model._accuracy[Bayesian_Model._keys[3]])
			data.append(Bayesian_Model._accuracy[Bayesian_Model._keys[0]])
			xlabel.append(str(observation) + "/ExpMech")
			xlabel.append(str(observation) + "/Laplace")

	plt.figure(figsize=(18,10))
	bplot = plt.boxplot(data, notch=1, widths=0.4, sym='+', vert=2, whis=1.5,patch_artist=True)
	plt.xlabel("different dimensions",fontsize=15)
	plt.ylabel('Accuracy / Hellinegr Distance',fontsize=15)
	#ax.set_xlim(0.5, len(errors) + 0.5)

	plt.xticks(range(1, len(data)+1),xlabel,rotation=70,fontsize=12)
	plt.title("Accuracy VS. Dimensionality",fontsize=20)
	print('Accuracy / prior: [1,1,...]' + ", delta: " + str(delta) + ", epsilon:" + str(epsilon) +  ', mean: uniform')
	for i in range(1, len(bplot["boxes"])/2 + 1):
		box = bplot["boxes"][2 * (i - 1)]
		box.set(color='navy', linewidth=1.5)
		box.set(facecolor='lightblue' )
		box = bplot["boxes"][2 * (i - 1) + 1]
		box.set(color='navy', linewidth=1.5)
		box.set(facecolor='darkkhaki' )
	plt.grid()
	plt.show()
	return

def accuracy_VS_prior_mean(sample_size,epsilon,delta,priors,observations):
	data = []
	xlabel = []
	for prior in priors:
		for observation in observations:
			Bayesian_Model = BayesInferwithDirPrior(prior, sample_size, epsilon, delta)
			Bayesian_Model._set_observation(observation)
			Bayesian_Model._experiments(300)
			data.append(Bayesian_Model._accuracy[Bayesian_Model._keys[3]])
			data.append(Bayesian_Model._accuracy[Bayesian_Model._keys[0]])
			xlabel.append(str(prior._alphas) + ", data:" + str(observation) + "/ExpMech")
			xlabel.append(str(prior._alphas) + ", data:" + str(observation) + "/Laplace")

	plt.figure(figsize=(18,10))
	bplot = plt.boxplot(data, notch=1, widths=0.4, sym='+', vert=2, whis=1.5,patch_artist=True)
	plt.xlabel("different prior distributions and different data set means")
	plt.ylabel('Accuracy / Hellinegr Distance')
	#ax.set_xlim(0.5, len(errors) + 0.5)

	plt.xticks(range(1, len(data)+1),xlabel,rotation=70)
	plt.title("Accuracy VS. Prior Distribution & Data Variance")
	print ('Accuracy / observation: ' + str(observation) + ", delta: " + str(delta) + ", epsilon:" + str(epsilon) +  ', mean:' + str(mean))
	for i in range(1, len(bplot["boxes"])/2 + 1):
		box = bplot["boxes"][2 * (i - 1)]
		box.set(color='navy', linewidth=1.5)
		box.set(facecolor='lightblue' )
		box = bplot["boxes"][2 * (i - 1) + 1]
		box.set(color='navy', linewidth=1.5)
		box.set(facecolor='darkkhaki' )
	plt.grid()
	plt.show()
	return

def gen_dataset(v, n):
	return [int(n * i) for i in v]

def gen_datasets(v, n_list):
	return [gen_dataset(v,n) for n in n_list]

# def gen_datasets(v, n):
# 	datasizes = 
# 	return [gen_dataset(v,n) for n in datasizes]

def gen_datasizes(r, step):
	return [i*step for i in range(r[0]/step,r[1]/step + 1)]

if __name__ == "__main__":

	datasize = 400
	epsilon = 1.0
	delta = 0.00000001
	prior = dirichlet([5000,5000,5000,5000])
	x1 = [1,19]
	x2 = [2,18]
	observation = [100,100,100,100]
	epsilons = numpy.arange(0.1, 2, 0.1)
	datasizes = gen_datasizes((7000,10000),1000)#[300] #[8,12,18,24,30,36,42,44,46,48]#,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80]
	percentage = [0.3,0.3,0.4]
	datasets = gen_datasets(percentage, datasizes)
	priors = [dirichlet([4*i,4*i,4*i]) for i in range(5,20)]
	# print opt_hellinger2([20000,20000],[19999, 19999])
	# accuracy_VS_dimension(sample_sizes, epsilon, delta)
	# accuracy_VS_prior_mean(sample_size,epsilon,delta,priors,observations)
	# means = [[(i/10.0), (1 - i/10.0)] for i in range(1,10)]
	# print means
	# accuracy_VS_prior(sample_size,epsilon,delta,priors,observation,mean)
	# accuracy_VS_mean(sample_size,epsilon,delta,prior)
	# accuracy_VS_datasize(epsilon,delta,prior,datasets,datasizes)
	# # hellinger_vs_l1norm(Dir(observation))
	# global_epsilon_study(sample_sizes,epsilon,delta,prior)
	# Dir([1,17]) - Dir([])
	# d0 = dirichlet([6,6])
	# d1 = dirichlet([10,2])
	# print d0._hellinger_sensitivity()
	# print d1._hellinger_sensitivity()

	# accuracy_VS_epsilon(sample_size,epsilons,delta,prior,observation)

	
	# epsilon_study(sample_size,epsilon,delta,prior,x1, x2)

	# print math.floor(-0.6)
	accuracy_study_discrete(datasize,epsilon,delta,prior,observation)
	# # accuracy_study_exponential_mechanism_SS(sample_size,epsilon,delta,prior,observation)
	# # accuracy_study_laplace(sample_size,epsilon,delta,prior,observation)
	# # Tests the functioning of the module

	# print Dir([10,10]) - Dir([1,19])

	# Bayesian_Model = BayesInferwithDirPrior(prior, sample_size, epsilon, delta)

	# Bayesian_Model._set_observation(observation)

	# Bayesian_Model._experiments(1000)

	# draw_error(Bayesian_Model._accuracy,Bayesian_Model, "order-2-size-30-runs-1000-epsilon-1.2-hellinger-delta000005-observation202020-box.png")

	# draw_error_l1(Bayesian_Model._accuracy_l1,Bayesian_Model, "order-2-size-100-runs-1000-epsilon-08-l1norm-delta000005box.png")
	

