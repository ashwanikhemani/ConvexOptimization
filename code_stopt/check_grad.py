import numpy as np, read_data, prob_grad
from scipy.optimize import check_grad

data = read_data.read_train_sgd()

def func(params, *args):
#computes function value for a single example
#note that parameters are not directly accessed
#instead they are manipulated through a view

	W, T = params[:26*129].reshape((26, 129)),\
		params[26*129:].reshape((26, 26))
	x, y = args[2]
	l = args[3]

	log_p = prob_grad.compute_log_p(x, y, W, T)
	
	return -1*log_p + 0.5*l*(\
		np.sum(np.square(np.linalg.norm(W, axis=1))) +\
		np.sum(np.square(T)))

log_grad = np.zeros(26*129+26*26)

l_gw, l_gt = log_grad[:26*129].reshape((26, 129)),\
	log_grad[26*129:].reshape((26, 26))

def func_prime(params, *args):
#computes the derivative of a single example
#note that parameters are not directly accessed
#instead they are manipulated through a view

	W, T = params[:26*129].reshape((26, 129)),\
		params[26*129:].reshape((26, 26))
	x, y = args[2]
	l = args[3]

	#compute first part of objective
	np.multiply(prob_grad.log_p_wgrad(W, x, y, T), -1, out=l_gw)
	np.multiply(prob_grad.log_p_tgrad(T, x, y, W), -1, out=l_gt)

	#add regularizers
	np.add(log_grad, np.multiply(l, params), out=log_grad)

	return log_grad

l = 0.5
params = np.random.rand((26*129+26*26))
W, T = params[:26*129].reshape((26, 129)),\
	params[26*129:].reshape((26, 26))

print(check_grad(func, func_prime, params, W, T, data[7], l))