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


class gaussian(object):
	def __init__(self, mean, variance):
		self._mean = mean
		self._variance = variance

	def _hellinger(self, other):
		return math.sqrt(1 - math.sqrt( 2 * math.sqrt(self._variance * other._variance) / (self._variance + other._variance)) 
			* math.exp(- 0.25 * (self._mean - other._mean) ** 2 / (self._variance + other._variance)))


	def __sub__(self, other):
		return self._hellinger(other)


	def _minus(self,other):
		self._alphas = list(numpy.array(self._alphas) - numpy.array(other._alphas))
		return self

	def __add__(self, other):
		return dirichlet(list(numpy.array(self._alphas) + numpy.array(other._alphas)))

	def show(self):
		print "Dirichlet("+str(self._alphas) + ")"

	def _hellinger_sensitivity(self):
		return

	def _adjacent(self, n):
		adjacents = []
		if self._mean - 1.0 / (1.0 + n) > 0.0:
			adjacents.append(gaussian(self._mean - 1.0 / (1.0 + n), self._variance))

		if self._mean + 1.0 / (1.0 + n) < 1.0:
			adjacents.append(gaussian(self._mean + 1.0 / (1.0 + n), self._variance))
		
		return adjacents

