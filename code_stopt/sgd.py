import numpy as np, read_data, prob_grad, random

norm = np.zeros(26)

#function computation
def func(params, *args):
#objective function specified in the handout
	W, T = params[:26*129].reshape((26, 129)), params[26*129:].reshape((26, 26))
	data = args[0]
	lmda = args[1]

	log_p = 0
	for example in data:
		log_p += prob_grad.compute_log_p(example[0], example[1], W, T)

	for i in range(26):
		norm[i] = np.linalg.norm(W[i])

	np.square(norm, out=norm)
	
	return -1*log_p/(len(data)) + lmda*0.5*(np.sum(norm)+ np.sum(np.square(T)))

log_grad_w = np.zeros(26*129).reshape((26, 129))
log_grad_t = np.zeros(26*26).reshape((26, 26))
grad = np.zeros(26*129+26*26)

def func_prime(params, *args):
#derivative of objective function specified in the handout
	W, T = params[:26*129].reshape((26, 129)), params[26*129:].reshape((26, 26))
	x, y = args[0]
	lmda = args[1]

	np.multiply(prob_grad.log_p_wgrad(W, x, y, T), -1, out=log_grad_w)
	np.multiply(prob_grad.log_p_tgrad(T, x, y, W), -1, out=log_grad_t)

	#add gradient of norm
	np.add(log_grad_w, W, out=log_grad_w)

	#add normalizing factor
	np.add(log_grad_t, T, out=log_grad_t)

	np.concatenate([log_grad_w.reshape(26*129),\
		log_grad_t.reshape(26*26)], out=grad)

#initial guess of 0
guess = np.zeros((26*129+26*26))
data = read_data.read_train_sgd()
learning_rate, l = 1e-5, 1e-2

#run sgd
for i in range(100):
	random.shuffle(data)
	print(func(guess, data, l))
	for example in data:
		func_prime(guess, example, l)
		np.multiply(learning_rate*-1, grad, out=grad)
		np.add(guess, grad, out=guess)
