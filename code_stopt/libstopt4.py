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
		
def sgd(path, init, lr, lmda):
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
	i, f = 0, open(path+f"/sgd-{lr}-{lmda}.txt" , "w")

	#momentum variable
	m = np.zeros(129*26+26*26, dtype=np.longdouble)

	#Run descent forever
	print(f"Starting SGD with Momentum: lr:{lr} lambda:{lmda}")
	print(f"Starting SGD with Momentum: lr:{lr} lambda:{lmda}", file=f)

	prev = 0.0
	while True:
		#compute decay rate
		temp_lr = lr/(1+0.5*i)

		#now check if we have converged print and return if the case
		current = func(guess, data, lmda)

		print(f"{i}:{current}:{temp_lr}", file=f)
		print(f"{i}\t{current}\t{temp_lr}")

		if abs(current - prev) < 1e-3:
			print("Convergence")
			return
		else:
			prev = current

		for j in range(len(data)):

			func_prime(guess, data[j], lmda)
			np.multiply(0.9, m, out=m)
			np.multiply(temp_lr, log_grad, out=log_grad)
			np.add(m, log_grad, out=m)
			np.subtract(guess, m, out=guess)

		i += 1

def compute_test_error(f, test_, W_, T_):

	letter_error, letter_count, word_error = 0.0, 0.0, 0.0
	for example in test_:
		letter_count += len(example[1])
		y_guess = max_sum(example[0], W_, T_)
		if not np.array_equal(y_guess, example[1]):
			word_error += 1
	return word_error/len(test_)

def adam(path, init, lr, lmda, epsilon):
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
	m_hat, v_hat = np.zeros(26*129+26*26), np.zeros(26*129+26*26)
	i, f = 0, open(path+f"/adam-{lr}-{lmda}.txt", "w")

	print(f"Running Adam: lr:{lr} lambda:{lmda} epsilon:{epsilon}")
	print(f"Running Adam: lr:{lr} lambda:{lmda} epsilon:{epsilon}", file=f)

	prev = 0.0
	while True:

		if t % 3438 == 0:
			random.shuffle(data)
			current = func(guess, data, lmda)
			print(f"{i}:{current}:{compute_test_error(f, test_data, W, T)}", file=f)
			print(f"{i}:{current}:{compute_test_error(f, test_data, W, T)}")
			if abs(current - prev) < 1e-3:
				print("Convergence")
				return
			else:
				prev = current

			i += 1
		t += 1

		temp_lr = lr/(1+0.5*i)

		func_prime(guess, data[t%len(data)], lmda)

		np.multiply(b1, m, out=m)
		np.add(m, np.multiply((1-b1), log_grad), out=m)

		np.multiply(b2, v, out=v)
		np.square(log_grad, out=log_grad)
		np.multiply((1-b2), log_grad, out=log_grad)
		np.add(v, log_grad, out=v)


		np.divide(m, (1-np.power(b1, t)), out=m_hat)
		np.divide(v, (1-np.power(b2, t)), out=v_hat)

		np.multiply(-1*temp_lr, m_hat,  out=m_hat)
		np.sqrt(v_hat, out=v_hat)
		np.add(v_hat, epsilon, out=v_hat)
		np.divide(m_hat, v_hat, out=m_hat)
		np.add(guess, m_hat, out=guess)

def sample(data, table, s):

	#pick word from training set
	X, y = random.choice(data)

	#pick random assignments for the labels
	y = np.random.randint(0, high=26, size=y.shape[0])

	#sample which is really slow
	probs = np.zeros(26)
	elements = np.arange(26)
	for i in range(s):

		indx = np.random.randint(0, high=y.shape[0])
		for j in range(26):
			y[indx] = j
			
		print(probs)
		y[indx] = np.random.choice(elements, 1, p=probs)

	#now use this to compute gradient
	return (X, y)

def compute_freq(data):
#position in the word matters here for the count

	table_node = np.zeros((26, 14))
	table_edge = np.zeros((26, 26))

	for example in data:
		table_node[example[1][0], 0] += 1
		for i in range(1, len(example[1])):
			table_node[example[1][i], i] += 1
			table_edge[example[1][i-1], example[1][i]] += 1
	
	np.divide(table, np.sum(table, axis=0), out=table)
	#compute row denom
	return table

def adam_mcmc(path, init, lr, lmda, epsilon, s):
#runs adam optimizer, inspired by ashwani

	print("Reading Train Data...")
	data = read_data.read_train_sgd()
	print("Reading Test Data...")
	test_data = read_data.read_test_sgd()

	print("Computing the frequencies")
	table = compute_freq(data)

	guess = np.copy(init)
	W, T = guess[:26*129].reshape((26, 129)),\
		guess[26*129:].reshape((26, 26))

	#adam parameters
	t, b1, b2, = 0, 0.9, 0.999
	m, v = np.zeros(26*129+26*26, dtype=np.longdouble), np.zeros(26*129+26*26, dtype=np.longdouble)
	i, f = 0, open(path+f"/adam-{lr}-{lmda}.txt", "w")

	print(f"Running Adam: lr:{lr} lambda:{lmda} epsilon:{epsilon}")
	print(f"Running Adam: lr:{lr} lambda:{lmda} epsilon:{epsilon}", file=f)

	prev = 0.0
	while True:

		if t % 30 == 0:
			current = func(guess, data, lmda)
			error = compute_test_error(f, test_data, W, T)
			print(f"{i}:{current}:{error}", file=f)
			print(f"{i}:{current}:{error}")
			if abs(current - prev) < 1e-3:
				print("Convergence")
				return
			else:
				prev = current

			i += 1
		t += 1

		temp_lr = lr/(1+0.5*i)

		example = sample(data, table,  s)
		func_prime(guess, example, lmda)

		np.multiply(b1, m, out=m)
		np.add(m, np.multiply((1-b1), log_grad), out=m)

		np.multiply(b2, v, out=v)
		np.square(log_grad, out=log_grad)
		np.multiply((1-b2), log_grad, out=log_grad)
		np.add(v, log_grad, out=v)

		np.divide(m, (1-np.power(b1, t)), out=m)
		np.divide(v, (1-np.power(b2, t)), out=v)

		np.multiply(-1*temp_lr, m, out=m)
		np.sqrt(v, out=v)
		np.add(v, epsilon, out=v)
		np.divide(m, v, out=m)
		np.add(guess, m, out=guess)




	
init = np.zeros((26*129+26*26), dtype=np.longdouble)
adam("output/", init, 1e-1, 1e-4, 1e-8)
#sgd(init, 1e-3, 1e-2)
#adam_mcmc("output", init, 1e-3, 1e-2, 1e-8, 14)
