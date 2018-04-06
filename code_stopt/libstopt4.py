import numpy as np, prob_grad, read_data, random
from scipy.optimize import check_grad
import random

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
		np.sum(np.square(W)) +\
		np.sum(np.square(T)))

def func1(params, *args):
#computes function value for a single example

	W, T = params[:26*129].reshape((26, 129)),\
		params[26*129:].reshape((26, 26))
	x, y = args[0]
	l = args[1]

	log_p = prob_grad.compute_log_p(x, y, W, T)
	
	return -1*log_p + 0.5*l*(\
		np.sum(np.square(W)) +\
		np.sum(np.square(T)))

#this is internal memory for func_prime
log_grad = np.zeros(26*129+26*26, dtype=np.longdouble)
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

	#I should return log_grad, but I am not for speed
	return log_grad

def max_sum(X, W, T):
#never called directly
#decodes by running the max sum algorithm
#X, W, T are numpy arrays (X is the input)
	alpha_len = 26
	trellis = np.zeros((X.shape[0],alpha_len))
	interior = np.zeros(alpha_len)
	y_star = np.zeros(X.shape[0], dtype=int)

	for i in range(1, X.shape[0]):
		for j in range(alpha_len):
			for k in range(alpha_len):
				interior[k] = np.dot(W[k], X[i-1]) +\
					T[k, j] + trellis[i-1, k]
			trellis[i, j] = np.max(interior)
	
	for i in range(alpha_len):
		interior[i] = np.dot(W[i], X[-1]) + trellis[-1, k]
	y_star[-1] = np.argmax(interior)

	for i in range(X.shape[0]-1, 0, -1):
		for j in range(alpha_len):
			interior[j] = np.dot(W[j], X[i-1]) +\
				T[j, y_star[i]] + trellis[i-1, j]
		y_star[i-1] = np.argmax(interior)

	return y_star

def compute_test_error(W, T):
	letter_error, letter_count, word_error = 0.0, 0.0, 0
	for example in test_data:
		letter_count += len(example[1])
		y_guess = max_sum(example[0], W, T)
		s = np.sum(y_guess != example[1])
		if s > 0:
			word_error += 1
		letter_error += s
	print(f"Letter Error {letter_error/letter_count},\
Word {word_error/len(test_data)}")
		
def sgd(init, lr, lmda):
#runs stochastic gradient descent on the function defined above
#starting at the intial guess of the params provided as an argument
#it also assumes the data you want to use is train_sgd, test_sgd
#outputs the letter and word wise error after ~1000 updates
#the below line is for testing if needed
#print(check_grad(func1, func_prime, guess, random.choice(data), lmda))
  
	print("Reading Train Data...")
	data = read_data.read_train_sgd()
	print("Reading Test Data...")
	test_data = read_data.read_test_sgd()

	guess = np.copy(init)
	W, T = guess[:26*129].reshape((26, 129)),\
		guess[26*129:].reshape((26, 26))

	#variables for printing to file
	i, f = 0, open("../results/S1/sgd-1e-2.txt", "w")

	#momentum variable
	m = np.zeros(129*26+26*26, dtype=np.longdouble)

	#Run descent forever
	print(f"Starting SGD with Momentum: lr:{lr} lambda:{lmda}")
	print(f"Starting SGD with Momentum: lr:{lr} lambda:{lmda}", file=f)
	while True:
		print(f"{i}:{func(guess, data, lmda)}:{lr}", file=f)
		print(f"{i}\t{func(guess, data, lmda)}\t{lr}")

		for j in range(len(data)):
			func_prime(guess, data[j], lmda)
			np.multiply(0.9, m, out=m)
			np.multiply(lr, log_grad, out=log_grad)
			np.add(m, log_grad, out=m)
			np.subtract(guess, m, out=guess)

		i += 1

def adam(init, lr, lmda, epsilon):
#runs adam optimizer, inspired by ashwani

	print("Reading Train Data...")
	data = read_data.read_train_sgd()
	print("Reading Test Data...")
	test_data = read_data.read_test_sgd()
	guess = np.copy(init)
	W, T = guess[:26*129].reshape((26, 129)),\
		guess[26*129:].reshape((26, 26))

	#adam parameters
	t, b1, b2, = 0, 0.9, 0.999
	m, v = np.zeros(26*129+26*26, dtype=np.longdouble), np.zeros(26*129+26*26, dtype=np.longdouble)
	i, f = 0, open("../results/S1/adam-1e-6.txt", "w")

	while True:

		if t % 3438 == 0:
			print(f"{i}:{func(guess, data, lmda)}")
			print(f"{i}:{func(guess, data, lmda)}", file=f)
			i += 1
		t += 1

		func_prime(guess, data[t%len(data)], lmda)

		np.multiply(b1, m, out=m)
		np.add(m, np.multiply((1-b1), log_grad), out=m)

		np.multiply(b2, v, out=v)
		np.square(log_grad, out=log_grad)
		np.multiply((1-b2), log_grad, out=log_grad)
		np.add(v, log_grad, out=v)

		np.divide(m, (1-np.power(b1, t)), out=m)
		np.divide(v, (1-np.power(b2, t)), out=v)

		np.multiply(-1*lr, m, out=m)
		np.sqrt(v, out=v)
		np.add(v, epsilon, out=v)
		np.divide(m, v, out=m)
		np.add(guess, m, out=guess)




init = np.zeros((26*129+26*26), dtype=np.longdouble)
sgd(init, 1e-3, 1e-2)
#adam(init, 1e-1, 1e-4, 1e-8)
