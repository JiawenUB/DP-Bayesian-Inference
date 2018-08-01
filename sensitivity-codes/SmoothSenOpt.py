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

def Hamming_Distance(dirichlet1, dirichlet2):
	temp = [abs(a - b) for a,b in zip(dirichlet1._alphas,dirichlet2._alphas)]
	return sum(temp)

def smooth_sensitivity_study(prior, sample_size, epsilon, delta, percentage):
	Bayesian_Model = BayesInferwithDirPrior(prior, sample_size, epsilon,delta)
	Bayesian_Model._set_observation([sample_size*i for i in percentage])

	Bayesian_Model._set_candidate_scores()
	Bayesian_Model._set_LS_Candidates()

	xstick = [str((c._minus(Bayesian_Model._prior))._alphas) for c in Bayesian_Model._candidates]
	beta = math.log(1 - epsilon / (2.0 * math.log(delta / (2.0 * (sample_size)))))
	y=[Bayesian_Model._LS_Candidates[r] * math.exp(- beta * Hamming_Distance(Bayesian_Model._posterior, r)) for r in Bayesian_Model._candidates]


	plot_ss(y,xstick,'Smooth Sensitivity Study w.r.t. x :' + str([sample_size*i for i in percentage]))


def plot_ss(ss,xstick,title):
	plt.figure(figsize=(12,10))
	plt.plot(range(len(ss)),ss,'r^')

	plt.xticks(range(len(ss)),xstick,rotation=70,fontsize=12)
	plt.title(title,fontsize=20)
	plt.xlabel(r"$y$",fontsize=25)	
	plt.ylabel(r"$\Delta_l(H(BI(y),-))e^{- \gamma *d(x,y)}$",fontsize=15)
	plt.grid()
	plt.legend()
	plt.show()

	return

if __name__ == "__main__":
	percentage = [0.5,0.5]
	datasize = 50
	prior = dirichlet([1,1])

	smooth_sensitivity_study(prior,datasize,0.8,0.00000001,percentage)

