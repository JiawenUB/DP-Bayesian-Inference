import numpy
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from copy import deepcopy
import random
import math
import scipy
from scipy.stats import beta
from mpl_toolkits.mplot3d import Axes3D
from dirichlet import dirichlet
from dpbayesinfer_Betabinomial import BayesInferwithDirPrior




def generate_sensitivities():
	sensitivities = []
	for i in range(1, 5000):
		j = 10000 - i
		sensitivities.append((dirichlet([i + 2, j]) - dirichlet([i + 1, j + 1]))/
			(dirichlet([i + 1, j + 1]) - dirichlet([i, j + 2])))

	return sensitivities


if __name__ == "__main__":

	sensitivities = generate_sensitivities()

	fig = plt.figure()

	plt.scatter(range(1, 5000), sensitivities)
	plt.xlabel(r'$\alpha, \beta = 10000 - \alpha$')
	plt.ylabel(r'$\frac{\mathcal{H}(\mathsf{beta}(\alpha+2, \beta),\mathsf{beta}(\alpha + 1, \beta+1))}{\mathcal{H}(\mathsf{beta}(\alpha + 1, \beta + 1), \mathsf{beta}(\alpha, \beta + 2))}$')
	plt.show()



	

