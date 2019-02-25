import math



def LAPLACE_CDF(interval, scale):
	if interval[0] >= 0.0:
		return (1 - 0.5 * math.exp( (-interval[1]*1.0/scale))) - (1 - 0.5 * math.exp( (-interval[0]/scale)))
	else:
		return (0.5 * math.exp( (interval[1]*1.0/scale))) - (0.5 * math.exp( (interval[0]/scale)))


if __name__ == "__main__":
	print (0.5 * (LAPLACE_CDF((1,2), 1.0)) + 0.5 * (LAPLACE_CDF((-1,0), 1.0)))




	

