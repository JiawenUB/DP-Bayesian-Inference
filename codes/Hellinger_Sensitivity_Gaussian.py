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
from gaussian import gaussian


def generate_sensitivities(adjacent_means, variance):
	sensitivities = []
	for i in range(len(adjacent_means) - 1):
		sensitivities.append(gaussian(adjacent_means[i], variance) - gaussian(adjacent_means[i+1], variance))
	return sensitivities


if __name__ == "__main__":

	means = [0.1 * i for i in range(10000)]

	variance = 1.0

	sensitivities = generate_sensitivities(means, variance)
	print sensitivities


	plt.figure()

	plt.plot(means[:-1], sensitivities)
	plt.xlabel(r'$\mu$')
	plt.ylabel(r'$\mathcal{H}(\mathcal{N}(\mu, \sigma ^ 2), \mathcal{N}(\mu + 0.1, \sigma ^ 2))$')
	
	plt.grid()
	plt.show()



	

