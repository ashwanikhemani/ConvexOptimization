import numpy, read_data, prob_grad
from scipy.optimize import check_grad

X_y = read_data.read_train_sgd()
W, T = numpy.random.rand(26*129).reshape(26, 129),\
	numpy.random.rand(26*26).reshape(26, 26)

def func(W, *args):
	w = W.reshape(26, 129)
	dataX = args[0]
	dataY = args[1]
	T = args[2]

	return prob_grad.compute_log_p(dataX, dataY, w, T)

def func_prime(W, *args):
	w = W.reshape(26, 129)
	dataX = args[0]
	dataY = args[1]
	T = args[2]
	
	return prob_grad.log_p_wgrad(w, dataX, dataY, T).reshape(26*129)

x0 = numpy.random.rand(26*129)

print(check_grad(func, func_prime, x0, X_y[0][0], X_y[0][1], T))
