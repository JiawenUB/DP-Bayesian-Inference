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


def row_discrete_probabilities(sample_size,epsilon,delta,prior,observation):
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
	probabilities_lap_1_by_steps = []
	probabilities_lap_2_by_steps = []

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

	# y = numpy.arange(0,4,1)
	laplace_probabilities_1 = {}
	laplace_probabilities_2 = {}
	for i in range(len(Bayesian_Model._candidates)):
		r = Bayesian_Model._candidates[i]
		t1 = 1.0
		t2 = 1.0
		# ylist = []
		for j in range(len(r._alphas) - 1):
			a = r._alphas[j] - Bayesian_Model._posterior._alphas[j]
			t1 = t1 * 0.5 * (math.exp(- ((abs(a)) if (a >= 0) else (abs(a) - 1)) / (2.0/epsilon)) - math.exp(- ((abs(a) + 1) if (a >= 0) else (abs(a))) / (2.0/epsilon)))
			t2 = t2 * 0.5 * (math.exp(- ((abs(a)) if (a >= 0) else (abs(a) - 1)) / (3.0/epsilon)) - math.exp(- ((abs(a) + 1) if (a >= 0) else (abs(a))) / (3.0/epsilon)))
			
		# 	ylist.append(a)
		# yset = set(ylist)
		laplace_probabilities_1[r] = t1 #/ (math.gamma(len(yset)) * (2 ** (len(list(filter(lambda a: a != 0, ylist))))))
		laplace_probabilities_2[r] = t2 #/ (math.gamma(len(yset)) * (2 ** (len(list(filter(lambda a: a != 0, ylist))))))

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

	# plt.figure()
	# plt.plot(steps[-100:], probabilities_exp_by_steps[-100:], 'ro', label=(r'$\mathcal{M}^{B}_{\mathcal{H}}$'))
	# # plt.plot(T, approximate_bounds, 'g^', label=('Expmech_SS zApproximate Bound'))
	# plt.plot(steps[-100:], probabilities_lap_1_by_steps[-100:], 'b^', label=('LapMech (sensitivity = 1)'))

	# plt.plot(steps[-100:], probabilities_lap_2_by_steps[-100:], 'gv', label=('LapMech (sensitivity = 2)'))

	# plt.xlabel("c / (steps from correct answer, in form of Hellinger Distance)")
	# plt.ylabel("Pr[H(BI(x),r) = c]")
	# plt.title("Discrete Probabilities")
	# plt.legend()
	# plt.grid()
	# plt.savefig("data_"+ str(observation) + "_prior_"+ str(Bayesian_Model._prior._alphas) + "_epsilon_"+ str(epsilon) + "_scatter.png")
	# plt.show()
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

	# plt.figure()
	# colors = ["ro","b^","gv"]
	# for i in range(len(filenames)):
	# 	plt.plot(steps[-100:], probabilities_by_steps[i][-100:], colors[i], label=(labels[i]))
	# 	# plt.plot(T, approximate_bounds, 'g^', label=('Expmech_SS zApproximate Bound'))
	# plt.xlabel("c / (steps from correct answer, in form of Hellinger Distance)")
	# plt.ylabel("Pr[H(BI(x),r) = c]")
	# plt.title("Discrete Probabilities")
	# plt.legend()
	# plt.grid()
	# plt.savefig(savename + "_scatter.png")
		# plt.show()
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
			if temp[0] == 0:
				temp[0] += 1
				temp[1] -= 1
			else:
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
		i = x2_probabilities[key]
		accuracy_epsilons[key] = math.log(item / i)

	sorted_epsilons = sorted(accuracy_epsilons.items(), key=operator.itemgetter(1))
	# print sorted_epsilons
	# print x1, x2
	# if sorted_epsilons[-1][1] > abs(sorted_epsilons[0][1]):
	# 	print sorted_epsilons[-1]
	# else:
	# 	print sorted_epsilons[0]
	# print max(sorted_epsilons[-1][1], abs(sorted_epsilons[0][1]))
	return max(sorted_epsilons[-1][1], abs(sorted_epsilons[0][1]))
	for key,value in sorted_epsilons:
		print "Pr[ ( M(x1) = " + key + ") / ( M(x2) = " + key + ") ] = exp(" + str(value) + ")"

	y = [value for key, value in sorted_epsilons]

	x = range(len(sorted_epsilons))

	xlabel = [key for key, value in sorted_epsilons]
	plt.figure(figsize=(15,8))
	plt.plot(x, y, 'bs-', label=('Exp Mech'))
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
		probabilities_exp[str(z._alphas)] = Bayesian_Model._SS_probabilities[i]
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


def plot_error_box(data, xlabel, xstick, title, legends, colors):
	l = len(legends)
	plt.figure(figsize=(0.5*len(data),9))
	medianprops = dict(linestyle='--', linewidth=3.0, color='green')
	# meanlineprops = dict(linestyle='--', linewidth=2.5, color='purple')
	bplot = plt.boxplot(data, notch=1, widths=0.4, sym='+', vert=2, whis=1.5, patch_artist=True, medianprops=medianprops)#, meanprops=meanlineprops, meanline=True,showmeans=True)
	plt.xlabel(xlabel,fontsize=15)
	plt.ylabel('Hellinegr Distance',fontsize=15)
	#ax.set_xlim(0.5, len(errors) + 0.5)


	plt.xticks([i*l + (l+1)/2.0 for i in range(len(xstick))],xstick,rotation=40,fontsize=12)
	plt.title(title,fontsize=20)

	for i in range(1, len(bplot["boxes"])/l + 1):
		for j in range(l):
			box = bplot["boxes"][l * (i - 1) + j]
			box.set(color=colors[j], linewidth=1.5)
			box.set(facecolor=colors[j] )
	plt.legend([bplot["boxes"][i] for i in range(l)], legends, loc='best')
	plt.grid()
	plt.show()

def plot_mean_error(x,y_list,xstick,xlabel, ylabel, title):
	plt.figure(figsize=(12,10))
	i = 0	
	for i in range(3):
		plt.plot(x,y_list[i],'^',label=ylabel[i])

	plt.xticks(x,xstick,rotation=70,fontsize=12)
	plt.title(title,fontsize=20)
	plt.xlabel(xlabel,fontsize=15)	
	plt.ylabel('Average Hellinger Distance ',fontsize=15)
	plt.grid()
	plt.legend()
	plt.show()

def accuracy_VS_datasize(epsilon,delta,prior,observations,datasizes):
	data = []
	xstick = []
	mean_error = [[],[],[]]
	for i in range(len(datasizes)):
		observation = observations[i]
		Bayesian_Model = BayesInferwithDirPrior(prior, sum(observation), epsilon, delta)
		Bayesian_Model._set_observation(observation)
		print("start" + str(observation))
		Bayesian_Model._experiments(3000)
		print("finished" + str(observation))

		# mean_error[0].append(Bayesian_Model._accuracy_mean[Bayesian_Model._keys[3]])
		# mean_error[1].append(Bayesian_Model._accuracy_mean[Bayesian_Model._keys[0]])
		# mean_error[2].append(Bayesian_Model._accuracy_mean[Bayesian_Model._keys[4]])
		data.append(Bayesian_Model._accuracy[Bayesian_Model._keys[3]])
		data.append(Bayesian_Model._accuracy[Bayesian_Model._keys[0]])
		data.append(Bayesian_Model._accuracy[Bayesian_Model._keys[4]])
		a = statistics.median(Bayesian_Model._accuracy[Bayesian_Model._keys[3]])
		b = statistics.median(Bayesian_Model._accuracy[Bayesian_Model._keys[0]])
		c = statistics.median(Bayesian_Model._accuracy[Bayesian_Model._keys[4]])
		# print("median: " + str(a) + ", " + str(b) + ", " + str(c))
		# if a == b:
		# 	print("the same")
		# else:
		# 	print("different")
		# xstick.append(str(datasizes[i]))
		# xstick.append(str(datasizes[i]))
		# xstick.append(str(datasizes[i]))
	print('Accuracy / prior: ' + str(prior._alphas) + ", delta: " 
		+ str(delta) + ", epsilon:" + str(epsilon))

	# plot_mean_error(range(0,len(observations)),mean_error,datasizes, 
	# 	 "Different Data sizes",
	# 	['Our ExpMech',"LapMech (sensitivity = 2)", "LapMech (sensitivity = 4)"],
	# 	"Mean Error VS. Data Size")
	plot_error_box(data,"Different Datasizes",datasizes,"Accuracy VS. Data Size",
		[r'$\mathcal{M}^{B}_{\mathcal{H}}$',"LapMech (sensitivity = 2)", "LapMech (sensitivity = 3)"],
		['lightblue', 'navy', 'red'])
	return




def accuracy_VS_prior(sample_size,epsilon,delta,priors,observation):
	data = []
	xstick = []
	xstick_mean = []
	mean_error = [[],[],[]]
	for prior in priors:
		Bayesian_Model = BayesInferwithDirPrior(prior, sample_size, epsilon, delta)
		Bayesian_Model._set_observation(observation)
		Bayesian_Model._experiments(1000)
		data.append(Bayesian_Model._accuracy[Bayesian_Model._keys[3]])
		data.append(Bayesian_Model._accuracy[Bayesian_Model._keys[0]])
		data.append(Bayesian_Model._accuracy[Bayesian_Model._keys[4]])
		mean_error[0].append(Bayesian_Model._accuracy_mean[Bayesian_Model._keys[3]])
		mean_error[1].append(Bayesian_Model._accuracy_mean[Bayesian_Model._keys[0]])
		mean_error[2].append(Bayesian_Model._accuracy_mean[Bayesian_Model._keys[4]])

		# xstick_mean.append(str(prior._alphas))

		# xstick.append(str(prior._alphas) + "/ExpMech")
		# xstick.append(str(prior._alphas) + "/Laplace")

	print('Accuracy / observation: ' + str(observation) + ", delta: " + str(delta) + ", epsilon:" + str(epsilon))
	
	# plot_mean_error(range(0,len(priors)),mean_error,xstick_mean,"Different Prior Distributions","Mean Accuracy VS. Prior Distribution")
	# plot_mean_error(range(0,len(priors)),mean_error,[r"$\mathsf{beta}$" + str(i._alphas) for i in priors], 
	# 	 r"Different Priors on $\theta$",
	# 	[r'$\mathcal{M}^{B}_{\mathcal{H}}$',"LapMech (sensitivity = 1)", "LapMech (sensitivity = 2)"],
	# 	"Accuracy VS. Prior Distribution")
	
	plot_error_box(data,r"Different Priors on $\theta$",[r"$\mathsf{beta}$" + str(i._alphas) for i in priors],
		"Accuracy VS. Prior Distribution",
		[r'$\mathcal{M}^{B}_{\mathcal{H}}$',"LapMech (sensitivity = 1)", "LapMech (sensitivity = 2)"],
		['navy', 'red', 'green'])
	return

def accuracy_VS_mean(sample_size,epsilon,delta,prior):
	data = []
	xstick = []
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
		xstick.append(str(observation._alphas) + "/ExpMech")
		xstick.append(str(observation._alphas) + "/Laplace")

	plot_error_box(data,"Different Prior Distributions",xstick,"Accuracy VS. Prior Distribution")

	return

def accuracy_VS_dimension(sample_sizes, epsilon, delta):
	data = []
	xstick = []
	for n in sample_sizes:
		for d in range(2,5):
			observation = [n for i in range(d)]
			prior = Dir([1 for i in range(d)])
			Bayesian_Model = BayesInferwithDirPrior(prior, n*d, epsilon, delta)
			Bayesian_Model._set_observation(observation)
			Bayesian_Model._experiments(500)
			data.append(Bayesian_Model._accuracy[Bayesian_Model._keys[3]])
			data.append(Bayesian_Model._accuracy[Bayesian_Model._keys[0]])
			xstick.append(str(observation) + "/ExpMech")
			xstick.append(str(observation) + "/Laplace")

	#ax.set_xlim(0.5, len(errors) + 0.5)

	plot_error_box(data,"Different Prior Distributions",xstick,"Accuracy VS. Prior Distribution")

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
			xstick.append(str(prior._alphas) + ", data:" + str(observation) + "/ExpMech")
			xstick.append(str(prior._alphas) + ", data:" + str(observation) + "/Laplace")

	plot_error_box(data,"Different Prior Distributions",xstick,"Accuracy VS. Prior Distribution")

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

def gen_priors(r, step, d):
	return [dirichlet([step*i for j in range(d)]) for i in range(r[0]/step,r[1]/step + 1)]


if __name__ == "__main__":

	datasize = 600
	epsilon = 1.0
	delta = 0.00000001
	prior = dirichlet([1,1])
	dataset = [300,300]
	x1 = [1,499]
	x2 = [0,500]
	epsilons = numpy.arange(0.1, 2, 0.1)
	datasizes = gen_datasizes((600,600),50)#[300] #[8,12,18,24,30,36,42,44,46,48]#,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80]
	percentage = [0.5,0.5]
	datasets = gen_datasets(percentage, datasizes)
	priors = [dirichlet([1,1])] + gen_priors([5,20], 5, 2) + gen_priors([40,100], 20, 2) + gen_priors([150,300], 50, 2) + gen_priors([400,400], 50, 2)
	# accuracy_VS_prior_mean(sample_size,epsilon,delta,priors,observations)
	# accuracy_VS_prior(datasize,epsilon,delta,priors,observation)
	# accuracy_VS_mean(sample_size,epsilon,delta,prior)
	# accuracy_VS_datasize(epsilon,delta,prior,datasets,datasizes)
	# hellinger_vs_l1norm(Dir(observation))
	# global_epsilon_study(datasizes,epsilon,delta,prior)
	# accuracy_VS_epsilon(sample_size,epsilons,delta,prior,observation)
	# for i in range(len(datasizes)):
	# 	row_discrete_probabilities(datasizes[i],epsilon,delta,prior,datasets[i])

	# row_discrete_probabilities(datasize,epsilon,delta,prior,dataset)
	# discrete_probabilities_from_file(
	# 	["datas/discrete_prob_3/exp/data_[198, 198, 204]_exp.txt",
	# 	"datas/discrete_prob_3/lap2/data_[198, 198, 204]_lap_sensitivity2.txt", 
	# 	"datas/discrete_prob_3/lap3/data_[198, 198, 204]_lap_sensitivity3.txt"],
	# 	[r'$\mathcal{M}^{B}_{\mathcal{H}}$',
	# 	"LapMech (sensitivity = 2)",
	# 	"LapMech (sensitivity = 3)"],
	# 	"poster_5")

	# epsilon_study(datasize,epsilon,delta,prior,x1, x2)
	# row_discrete_probabilities(datasize,epsilon,delta,prior,observation)



	
	# # accuracy_study_exponential_mechanism_SS(sample_size,epsilon,delta,prior,observation)
	# # accuracy_study_laplace(sample_size,epsilon,delta,prior,observation)

	

