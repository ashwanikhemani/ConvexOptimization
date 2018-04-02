import numpy as np, prob_grad, read_data, random

#this is internal memory for func_prime
log_grad = np.zeros(26*129+26*26)
l_gw, l_gt = log_grad[:26*129].reshape((26, 129)),\
	log_grad[26*129:].reshape((26, 26))

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

test_data = read_data.read_test_sgd()[:100]

def compute_test_error(W, T):
	letter_error, letter_count, word_error = 0.0, 0.0, 0
	for example in test_data:
		letter_count += len(example[1])
		y_guess = max_sum(example[0], W, T)
		s = np.sum(y_guess != example[1])
		if s > 0:
			word_error += 1
		letter_error += s
	print(f"Letter Error {letter_error/letter_count}, Word {word_error/len(test_data)}")
		
def sgd(init, lr, lmda):
#runs stochastic gradient descent on the function defined above
#it also assumes the data you want to use
#starting at the intial guess of the params provided as an argument
  
	data = read_data.read_train_sgd()
	#create views of copy of init
	guess = np.copy(init)
	W, T = guess[:26*129].reshape((26, 129)),\
		guess[26*129:].reshape((26, 26))

	compute_test_error(W, T)
	for i in range(100):
		random.shuffle(data)
		for j in range(len(data)):
			func_prime(guess, data[j], lmda)
			np.multiply(lr*-1, log_grad, out=log_grad)
			np.add(guess, log_grad, out=guess)
			if j % 1000 == 0:
				compute_test_error(W, T)

init = np.zeros((26*129+26*26))
sgd(init, 1e-3, 1e-3)
