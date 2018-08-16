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
from scipy.integrate import quad

def beta_function(alpha, beta):
	return 1.0 * math.gamma(alpha) * math.gamma(beta)/math.gamma(alpha + beta)

def Hellinger_Distance(beta1, beta2):
	return math.sqrt(1 - beta_function((beta1[0] + beta2[0])/2.0, (beta1[1] + beta2[1])/2.0) \
		/ math.sqrt(beta_function(beta1[0], beta1[1]) * beta_function(beta2[0], beta2[1])))

class Beta_Distribution(object):
	def __init__(self, alpha, beta):
		self._alpha = alpha
		self._beta = beta

	def __sub__(self, other):
		# return Hellinger_Distance((self._alpha, self._beta), (other._alpha, other._beta))
		return KL_Distance(self, other)


	def show(self):
		print "Beta(" + str(self._alpha) + ", " + str(self._beta) + ")"

	def pdf(self, x):
		return x ** (self._alpha - 1) * (1 - x) ** (self._beta - 1) / beta_function(self._alpha, self._beta)

def KL_Distance_point(x, beta1, beta2):
	return beta1.pdf(x) * math.log(beta2.pdf(x) / beta1.pdf(x))

def KL_Distance(beta1, beta2):
	return 0 - quad(KL_Distance_point, 0, 1, args=(beta1,beta2))[0]



def generate_sensitivities():
	sensitivities = []
	for i in range(2, 82):
		sensitivities.append([])
		for j in range(2, 82):
			sensitivities[i - 2].append(Beta_Distribution(i + 1, j) - Beta_Distribution(i, j + 1))
			# print sensitivities[i - 1][j - 1]
	return sensitivities


if __name__ == "__main__":

	sensitivities = generate_sensitivities()

	fig = plt.figure()
	ax = fig.gca(projection='3d')
	# x = numpy.arange(1, 81, 1)
	# y = numpy.arange(1, 81, 1)
	# x, y = numpy.meshgrid(x,y)
	print sensitivities
	x = []
	y = []
	z = []
	for i in range(2, 82):
		x = x + [i] * 80
		y = y + range(2, 82)
		z = z + sensitivities[i - 2]

	#surf = ax.plot_surface(x, y, numpy.array(sensitivities), cmap=cm.coolwarm, linewidth=0, antialiased=False)
	#fig.colorbar(surf, shrink=0.5, aspect=5)
	ax.scatter(x, y, z)
	ax.set_xlabel('Alpha')
	ax.set_ylabel('Beta')
	ax.set_zlabel('KL(beta(a+1, b), Beta(a, b+1))')
	plt.show()



	

