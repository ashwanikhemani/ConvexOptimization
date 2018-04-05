import numpy as np, read_data, prob_grad, random
from scipy.optimize import check_grad

l = 10
data = read_data.read_train_sgd()

def func(params, *args):
#computes function value for a single example

	W, T = params[:26*129].reshape((26, 129)),\
		params[26*129:].reshape((26, 26))
	l = args[1]
	return  0.5*l*(\
		np.sum(np.square(W)) +\
		np.sum(np.square(T)))

def func_prime(params, *args):
#computes the derivative of a single example

	W, T = params[:26*129].reshape((26, 129)),\
		params[26*129:].reshape((26, 26))
	l = args[1]

	#add regularizers
	log_grad = np.zeros(26*129+26*26)
	np.add(log_grad, np.multiply(l, params), out=log_grad)

	return log_grad

#params = np.random.rand((26*129+26*26))
params = np.ones(26*129+26*26)

print(check_grad(func, func_prime, params, random.choice(data), l))
