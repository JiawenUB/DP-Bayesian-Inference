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
		return math.sqrt(1.0 - math.sqrt(2.0 * other._variance * self._variance / (other._variance ** 2 + self._variance ** 2)) * math.exp( - 0.25 * ( self._mean - other._mean) ** 2 / (other._variance ** 2 + self._variance ** 2)))


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

