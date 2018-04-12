#script that does everything related to sgd

import sys

path_train = sys.argv[1]
path_test = sys.argv[2]
path_output = sys.argv[3]
lr = 1e-2 
lmda = 1e-2
tol = 1e-3


from string import ascii_lowercase

import numpy as np

def read_data(path):
#reads data from path and returns a list of tuples labels are 0-25
#assumes the same format for each file
#space separated, 2nd is label (as a character) 3rd is position
#from 6th on is the raw data
#max word length is 14

	print(f"Reading data from: {path}")
	mapping = list(enumerate(ascii_lowercase))
	mapping = { i[1]:i[0] for i in mapping }

	with open(path, "r") as f:
		raw_data = f.read()
	raw_data = raw_data.split("\n")

	dataX, dataY = [], []
	tempX, tempY = [], []
	for row in raw_data[:-1]:
		row = row.split(" ")
		tempY.append( mapping[row[1]])
		tempX.append( np.concatenate(\
			( np.array(row[5:], dtype=float), np.array([1]) )))
		if int(row[2]) < 0:
			dataX.append(np.array(tempX))
			dataY.append(np.array(tempY, dtype=int))
			tempX, tempY = [], []

	ret = zip(dataX, dataY)
	return list(ret)

#read the training data
train_data = read_data(path_train)
test_data = read_data(path_test)

#create param memory
param_guess = np.zeros(26*(26+129), dtype=float)

def compute_log_p(X, y, W_, T_):
#define how to compute log probability
#returns the log probability of a set of labels given X
#the parameters should all be numpy arrays of some kind
#I assume the labels are all shifted to the left by one
	alpha_len = 26 #I would like to make this a parameter for generality
			
	sum_num = np.dot(W_[y[0]], X[0])
	for i in range(1, X.shape[0]):
		sum_num += np.dot(W_[y[i]], X[i]) + T_[y[i-1], y[i]]
	
	trellisfw = np.zeros((X.shape[0], alpha_len), dtype=float)
	interior = np.zeros(alpha_len, dtype=float)
	messages = np.zeros((26, 26), dtype=float)

	for i in range(1, X.shape[0]):
		np.matmul(W_, X[i-1], out=interior)
		np.add(interior, trellisfw[i-1], out=interior)
		np.add(T_, interior[:, np.newaxis], out=messages)
		maxes = messages.max(axis=0)
		np.add(messages, -1*maxes, out=messages)
		np.exp(messages, out=messages)
		np.sum(messages, axis=0, out=interior)
		np.log(interior, out=interior)
		np.add(maxes, interior, out=trellisfw[i])

	dots = np.matmul(W_, X[-1])
	np.add(dots, trellisfw[-1], out=interior)

	M = np.max(interior)
	np.add(interior, -1*M, out=interior)
	np.exp(interior, out=interior)
	
	log_z = M + np.log(np.sum(interior))

	return sum_num - log_z

def func(params, *args):
#define the objective function as specified in the writeup
#computes function value for a single example

	data, W_, T_, lmda_ = args[0], args[1], args[2], args[3]

	log_p = 0
	for example in data:
		log_p += compute_log_p(example[0], example[1], W_, T_)
	
	return -1*log_p/len(data) + 0.5*lmda_*(\
		np.sum(np.square(W_)) +\
		np.sum(np.square(T_)))

def fb_prob(X, W_, T_):
#runs the forward backward algorithm and is important for computing
#marginal probabilities
	alpha_len = 26

	trellisfw = np.zeros((X.shape[0], alpha_len), dtype=float)
	trellisbw = np.zeros((X.shape[0], alpha_len), dtype=float)

	interior = np.zeros(alpha_len, dtype=float)
	messages = np.zeros((26, 26), dtype=float)

	#forward part
	for i in range(1, X.shape[0]):
		np.matmul(W_, X[i-1], out=interior)
		np.add(interior, trellisfw[i-1], out=interior)
		np.add(T_, interior[:, np.newaxis], out=messages)
		maxes = messages.max(axis=0)
		np.add(messages, -1*maxes, out=messages)
		np.exp(messages, out=messages)
		np.sum(messages, axis=0, out=interior)
		np.log(interior, out=interior)
		np.add(maxes, interior, out=trellisfw[i])
		
	dots = np.matmul(W_, X[-1])
	np.add(dots, trellisfw[-1], out=interior)
	M = np.max(interior)
	np.add(interior, -1*M, out=interior)
	np.exp(interior, out=interior)
	
	log_z = M + np.log(np.sum(interior))

	#backward part
	for i in range(X.shape[0]-2, -1, -1):
		np.matmul(W_, X[i+1], out=interior)
		np.add(interior, trellisbw[i+1], out=interior)
		np.add(T_, interior, out=messages)
		np.swapaxes(messages, 0, 1)
		maxes = messages.max(axis=1)
		np.add(messages, -1*maxes[:, np.newaxis], out=messages)
		np.exp(messages, out=messages)
		np.sum(messages, axis=1, out=interior)
		np.log(interior, out=interior)
		np.add(maxes, interior, out=trellisbw[i])

	return trellisfw, trellisbw, log_z

def log_p_wgrad(W_, X, y, T_):
#will compute the gradient for the nodes of an example

	grad = np.zeros((26, 129),dtype=float) #size of the alphabet by 128 elems
	expect = np.zeros(26, dtype=float)
	trellisfw, trellisbw, log_z = fb_prob(X, W_, T_)
	prob = np.zeros(26, dtype=float)

	for i in range(X.shape[0]):
		#compute the marginal probability for all nodes in this column
		np.add(trellisfw[i, :], trellisbw[i, :], out=prob)
		np.add(np.matmul(W_, X[i]), prob, out=prob)
		np.add(-1*log_z, prob, out=prob)
		np.exp(prob, out=prob)

		#compute the expectation?
		expect[y[i]] = 1
		np.add(expect, -1*prob, out=expect)
		letter_grad = np.tile(X[i], (26, 1))
		#compute letter wise probabililty
		np.multiply(expect[:, np.newaxis], letter_grad,\
			out=letter_grad)
		np.add(grad, letter_grad, out=grad)
		expect[:] = 0

	return grad

def log_p_tgrad(T_, X, y, W_):
#will compute the edge gradient of an example

	grad = np.zeros((26, 26), dtype=float)
	potential = np.zeros((26, 26), dtype=float)
	expect = np.zeros((26, 26), dtype=float)
	trellisfw, trellisbw, log_z = fb_prob(X, W_, T_)

	for i in range(X.shape[0]-1):
		np.add.outer(np.matmul(W_, X[i]), np.matmul(W_, X[i+1]),\
			out=potential)
		np.add(T_, potential, out=potential)
		np.add(trellisfw[i][:, np.newaxis], potential, out=potential)
		np.add(trellisbw[i+1], potential, out=potential)
		np.add(-1*log_z, potential, out=potential)
		np.exp(potential, out=potential)
		expect[y[i], y[i+1]] = 1
		np.add(expect, -1*potential, out=potential)
		np.add(grad, potential, out=grad)
		expect[:, :] = 0
	return grad

#create memory for f' and a view
log_grad = np.zeros(26*(26+129), dtype=float)
l_gw, l_gt = log_grad[:26*129].reshape((26, 129)),\
	log_grad[26*129:].reshape((26, 26))

def func_prime(params, *args):
#defines the way to compute the gradient given a single example
#returns a numpy array which is that gradient

	x, y, W_, T_, lmda_ = args[0], args[1], args[2], args[3], args[4]

	#compute first part of objective
	np.multiply(log_p_wgrad(W_, x, y, T_), -1, out=l_gw)
	np.multiply(log_p_tgrad(T_, x, y, W_), -1, out=l_gt)

	#add regularizers
	np.add(log_grad, np.multiply(lmda_, params), out=log_grad)

	return log_grad

def max_sum(X, W_, T_):
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
				interior[k] = np.dot(W_[k], X[i-1]) +\
					T_[k, j] + trellis[i-1, k]
			trellis[i, j] = np.max(interior)
	
	for i in range(alpha_len):
		interior[i] = np.dot(W_[i], X[-1]) + trellis[-1, k]
	y_star[-1] = np.argmax(interior)

	for i in range(X.shape[0]-1, 0, -1):
		for j in range(alpha_len):
			interior[j] = np.dot(W_[j], X[i-1]) +\
				T_[j, y_star[i]] + trellis[i-1, j]
		y_star[i-1] = np.argmax(interior)

	return y_star

def compute_test_error(f, W_, T_):
	letter_error, letter_count, word_error = 0.0, 0.0, 0.0
	for example in test_data:
		letter_count += len(example[1])
		y_guess = max_sum(example[0], W_, T_)
		s = np.sum(y_guess != example[1])
		if not np.array_equal(y_guess, example[1]):
			word_error += 1
	return word_error/len(test_data)

def sgd_momentum_decay(path, train_, test_, params, lr_, lmda_, tol_):
#runs stochastic gradient descent on the function defined above
#starting at the intial guess of the params provided as an argument
#it also assumes the data you want to use is train_sgd, test_sgd
#outputs the letter and word wise error after ~1000 updates
#the below line is for testing if needed
  
	guess = np.copy(params)
	W_, T_ = guess[:26*129].reshape((26, 129)),\
		guess[26*129:].reshape((26, 26))

	#variables for printing to file
	i, f = 0, open(path+f"/sgd-{lr_}-{lmda_}.txt" , "w")

	#decay variable
	decay = 0.5

	#momentum variable
	m = np.zeros(129*26+26*26, dtype=float)

	#Run descent forever
	print(f"Starting SGD with Momentum and Decay: lr:{lr_} lambda:{lmda_} tol:{tol}")
	print(f"Starting SGD with Momentum and Decay: lr:{lr_} lambda:{lmda_} tol:{tol}", file=f)

	prev = 0.0
	while True:
		#compute decay rate
		temp_lr = lr_/(1+decay*i)

		#now check if we have converged print and return if the case
		current = func(guess, train_, W_, T_, lmda_)

		print(f"{i}:ObjVal: {current} LR: {temp_lr} ", file=f)
		print(f"{i}\t{current}\t{temp_lr}")

		if abs(current - prev) < tol_:
			print("Convergence")
			return
		else:
			prev = current

		for j in range(len(train_)):

			func_prime(guess, train_[j][0], train_[j][1],W_, T_, lmda_)
			np.multiply(0.9, m, out=m)
			np.multiply(temp_lr, log_grad, out=log_grad)
			np.add(m, log_grad, out=m)
			np.subtract(guess, m, out=guess)

		i += 1

def sgd_momentum(path, train_, test_, params, lr_, lmda_, tol_):
#runs stochastic gradient descent on the function defined above
#starting at the intial guess of the params provided as an argument
#it also assumes the data you want to use is train_sgd, test_sgd
#outputs the letter and word wise error after ~1000 updates
#the below line is for testing if needed
  
	guess = np.copy(params)
	W_, T_ = guess[:26*129].reshape((26, 129)),\
		guess[26*129:].reshape((26, 26))

	#variables for printing to file
	i, f = 0, open(path+f"/sgd-{lr_}-{lmda_}.txt" , "w")

	#momentum variable
	m = np.zeros(129*26+26*26, dtype=float)

	#Run descent forever
	print(f"Starting SGD with Momentum: lr:{lr_} lambda:{lmda_} tol:{tol}")
	print(f"Starting SGD with Momentum: lr:{lr_} lambda:{lmda_} tol:{tol}", file=f)

	prev = 0.0
	while True:
		#now check if we have converged print and return if the case
		current = func(guess, train_, W_, T_, lmda_)

		print(f"{i}:{current}", file=f)
		print(f"{i}\t{current}")
		compute_test_error(f, W_, T_)

		if abs(current - prev) < tol_:
			print("Convergence")
			return
		else:
			prev = current

		for j in range(len(train_)):

			func_prime(guess, train_[j][0], train_[j][1],W_, T_, lmda_)
			np.multiply(0.9, m, out=m)
			np.multiply(lr_, log_grad, out=log_grad)
			np.add(m, log_grad, out=m)
			np.subtract(guess, m, out=guess)

		i += 1

def sgd(path, train_, test_, params, lr_, lmda_, tol_):
#runs stochastic gradient descent on the function defined above
#starting at the intial guess of the params provided as an argument
#it also assumes the data you want to use is train_sgd, test_sgd
#outputs the letter and word wise error after ~1000 updates
#the below line is for testing if needed
  
	guess = np.copy(params)
	W_, T_ = guess[:26*129].reshape((26, 129)),\
		guess[26*129:].reshape((26, 26))

	#variables for printing to file
	i, f = 0, open(path+f"/sgd-{lr_}-{lmda_}.txt" , "w")

	#Run descent forever
	print(f"Starting SGD: lr:{lr_} lambda:{lmda_} tol:{tol}")
	print(f"Starting SGD: lr:{lr_} lambda:{lmda_} tol:{tol}", file=f)

	prev = 0.0
	print("iter\tObjective\tWordErr")
	print("iter\tObjective\tWordErr", file=f)
	while True:

		#now check if we have converged print and return if the case
		current = func(guess, train_, W_, T_, lmda_)

		print(f"{i}\t{current}\t{compute_test_error(f, W_, T_)}", file=f)
		print(f"{i}\t{current}\t{compute_test_error(f, W_, T_)}")

		if abs(current - prev) < tol_:
			print("Convergence")
			return
		else:
			prev = current

		for j in range(len(train_)):

			func_prime(guess, train_[j][0], train_[j][1],W_, T_, lmda_)
			np.multiply(-1*lr_, log_grad, out=log_grad)
			np.add(guess, log_grad, out=guess)

		i += 1

def sgd_decay(path, train_, test_, params, lr_, lmda_, tol_):
#runs stochastic gradient descent on the function defined above
#starting at the intial guess of the params provided as an argument
#it also assumes the data you want to use is train_sgd, test_sgd
#outputs the letter and word wise error after ~1000 updates
#the below line is for testing if needed
  
	guess = np.copy(params)
	W_, T_ = guess[:26*129].reshape((26, 129)),\
		guess[26*129:].reshape((26, 26))

	#variables for printing to file
	i, f = 0, open(path+f"/sgd-{lr_}-{lmda_}.txt" , "w")

	#decay variable
	decay = 0.5

	#Run descent forever
	print(f"Starting SGD with decay: lr:{lr_} lambda:{lmda_} tol:{tol}")
	print(f"Starting SGD with decay: lr:{lr_} lambda:{lmda_} tol:{tol}", file=f)

	print("iter\tlr\tObjective\tWordErr")
	print("iter\tlr\tObjective\tWordErr", file=f)

	prev = 0.0
	while True:

		#compute decay rate
		temp_lr = lr_/(1+decay*i)

		#now check if we have converged print and return if the case
		current = func(guess, train_, W_, T_, lmda_)

		print(f"{i}\t{temp_lr}\t{current}\t{compute_test_error(f, W_, T_)}", file=f)
		print(f"{i}\t{temp_lr}\t{current}\t{compute_test_error(f, W_, T_)}")

		if abs(current - prev) < tol_:
			print("Convergence")
			return
		else:
			prev = current

		for j in range(len(train_)):

			func_prime(guess, train_[j][0], train_[j][1],W_, T_, lmda_)
			np.multiply(-1*temp_lr, log_grad, out=log_grad)
			np.add(guess, log_grad, out=guess)

		i += 1


sgd_decay(path_output, train_data, test_data, param_guess, lr, lmda, tol)
