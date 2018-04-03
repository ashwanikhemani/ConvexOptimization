import numpy as np, read_data, prob_grad, random

norm = np.zeros(26)

def func(params, *args):
#computes function value for a single example

	W, T = params[:26*129].reshape((26, 129)),\
		params[26*129:].reshape((26, 26))
	data = args[0]
	l = args[1]

	log_p = 0
	for example in data:
		log_p += prob_grad.compute_log_p(example[0], example[1], W, T)
	
	return -1*log_p/len(data) + 0.5*l*(\
		np.sum(np.square(np.linalg.norm(W, axis=1))) +\
		np.sum(np.square(T)))


log_grad = np.zeros(26*129+26*26)

l_gw, l_gt = log_grad[:26*129].reshape((26, 129)),\
	log_grad[26*129:].reshape((26, 26))

def func_prime(params, *args):
#computes the derivative of a single example

    W, T = params[:26*129].reshape((26, 129)),\
		params[26*129:].reshape((26, 26))
    x, y = args[0]
    l = args[1]

	#compute first part of objective
    np.multiply(prob_grad.log_p_wgrad(W, x, y, T), -1, out=l_gw)
    np.multiply(prob_grad.log_p_tgrad(T, x, y, W), -1, out=l_gt)

	#add regularizers
    np.add(log_grad, np.multiply(l, params), out=log_grad)

    return log_grad

#initial guess of 0
guess = np.zeros((26*129+26*26))
data = read_data.read_train_sgd()
learning_rate, l = 1e-3, 1e-2

##run sgd
#for i in range(100):
#	random.shuffle(data)
#	if(i % 5 == 0):
#		print(func(guess, data, l))
#	for example in data:
#		func_prime(guess, example, l)
#		np.multiply(learning_rate*-1, log_grad, out=log_grad)
#		np.add(guess, log_grad, out=guess)
