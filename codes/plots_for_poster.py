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
import glob
import os

def plot_error_box(data, xlabel, xstick, title, legends, colors):
	l = len(legends)
	plt.figure(figsize=(0.5*len(data),9))
	medianprops = dict(linestyle='--', linewidth=3.0, color='green')
	# meanlineprops = dict(linestyle='--', linewidth=2.5, color='purple')
	bplot = plt.boxplot(data, notch=1, widths=0.4, sym='+', vert=2, whis=1.5, patch_artist=True, medianprops=medianprops)#, meanprops=meanlineprops, meanline=True,showmeans=True)
	plt.xlabel(xlabel,fontsize=15)
	plt.ylabel('Hellinegr Distance',fontsize=15)
	#ax.set_xlim(0.5, len(errors) + 0.5)


	plt.xticks([i*l + (l+1)/2.0 for i in range(len(xstick))],xstick,rotation=70,fontsize=12)
	plt.title(title,fontsize=20)

	for i in range(1, len(bplot["boxes"])/l + 1):
		for j in range(l):
			box = bplot["boxes"][l * (i - 1) + j]
			box.set(color=colors[j], linewidth=1.5)
			box.set(facecolor=colors[j] )
	plt.legend([bplot["boxes"][i] for i in range(l)], legends, loc='best')
	plt.grid()
	plt.show()

def percentile_box_plot(folders, percentile, legends):
	steps_of_persentile = []
	xsticks = []

	for  folder in folders:
		step_of_percentile_of_mech = []
		stick = []
		for file in glob.glob(os.path.join(folder, '*.txt')):
			stick.append(file.split("_")[2])
			expected_prob = []
			step = []
			print file
			f = open(file, "r")
			f.readline()
			for line in f:
				l = line.strip("\n").split("&")
				expected_prob.append(float(l[2]) * float(l[1]))
			# print prob
			f.close()
			step_of_percentile_of_mech.append(expected_prob)
		xsticks = stick
		steps_of_persentile.append(step_of_percentile_of_mech)

	data = []

	for mech in steps_of_persentile:
		for prob in mech:
			data.append(prob)

	x = numpy.arange(len(steps_of_persentile[0]))
	
	plot_error_box(data, "Different data sets", xsticks, "4 - Quantile of Hellinger Distance",legends, ['lightblue', 'navy', 'red'] )

	return	



def percentile_plot(folders, percentile, labels, savename):
	steps_of_persentile = []
	xsticks = []

	for  folder in folders:
		step_of_percentile_of_mech = []
		stick = []
		for file in glob.glob(os.path.join(folder, '*.txt')):
			stick.append(file.split("_")[2])
			prob = []
			step = []
			f = open(file, "r")
			f.readline()
			for line in f:
				l = line.strip("\n").split("&")
				prob.append(float(l[2]))
				step.append(float(l[1]))
			# print prob
			f.close()
			print file
			prob = numpy.array(prob)
			pcen=numpy.percentile(prob,percentile)
			i_near=abs(prob-pcen).argmin()
			step_of_percentile_of_mech.append(step[i_near])
		xsticks = stick
		steps_of_persentile.append(step_of_percentile_of_mech)

	x = numpy.arange(len(steps_of_persentile[0]))
	
	plt.figure()
	colors = ["r","b","g"]

	for i in range(len(folders)):
		plt.plot(x, steps_of_persentile[i], colors[i], label=(labels[i]))


	# for i in range(len(filenames)):
		# plt.plot(T, approximate_bounds, 'g^', label=('Expmech_SS zApproximate Bound'))
	plt.xlabel("Different data set observed")
	plt.ylabel("Hellinger Distance")
	plt.title("Accuracy of 60 Percentile")
	plt.legend()
	plt.grid()
	plt.xticks(x,xsticks, rotation=70,fontsize=8)

	plt.show()
	plt.savefig(savename + "_line.png")

	return	

if __name__ == "__main__":
	percentile_box_plot(["datas/discrete_prob/exp/",
		"datas/discrete_prob/lap2/","datas/discrete_prob/lap3/"],
		60, ["Our ExpMech", "LapMech (Sensitivity = 2)", 
		"LapMech (sensitivity = dimension)"])

	

