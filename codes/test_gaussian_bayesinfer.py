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


def gen_data(n, value = 0.0):
	if value == 0:
		return [random.random() for i in range(n)]
	else:
		return [value for i in range(n)]

if __name__ == "__main__":

	datasize = 99
	epsilon = 1.0
	delta = 0.00000001
	prior = gaussian(0.0, 1.0)
	known_variance = 1.0
	data = gen_data(datasize, 0.1)

	DP_model = DP_Bayesian_Inference_Gaussian(Bayesian_Inference_Gaussian(prior,data, known_variance), epsilon, delta)
	DP_model._update_model_setting()

	for i in range(100):
		r = DP_model._exponential_mechanism()
		print "EXPONENTIAL MECHANISM RESULT: GAUSSIAN(" + str( (r._mean, r._variance)) + " DISTANCE: " + str(r - DP_model._infer_model._posterior)


	for i in range(100):
		r = DP_model._laplace_mechanism()
		print "LAPLACE MECHANISM RESULT: GAUSSIAN(" + str( (r._mean, r._variance)) + " DISTANCE: " + str(r - DP_model._infer_model._posterior)
