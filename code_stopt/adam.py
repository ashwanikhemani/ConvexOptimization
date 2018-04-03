# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 10:18:58 2018

@author: ashwa
"""
import numpy as np, read_data,sgd


def adam_gd():
    log_grad = np.zeros(26*129+26*26)
    m_grad = np.zeros(26*129+26*26)
    v_grad = np.zeros(26*129+26*26)
    t=0    
    alpha=0.001
    beta1=0.9
    beta2=0.999
    epsilon=1e-10
    max_iter=3000
    tol=1e-6
    #initial guess of 0
    guess = np.zeros((26*129+26*26))
    data = read_data.read_train_sgd()
    l = 1e-2
    print('Running ADAM')
    while True :
        temp = np.zeros((26*129+26*26))
        guess_new = np.zeros((26*129+26*26))
        m_grad_new = np.zeros(26*129+26*26)
        v_grad_new = np.zeros(26*129+26*26)
#        print('Iteration '+str(t))
        t=t+1
        if(t % 5 == 0):
            print(sgd.func(guess, data, l))
        for example in data:
    #        Get gradients w.r.t. stochastic objective at timestep t)
            log_grad=sgd.func_prime(guess,example,l)
    #        Update biased first moment estimate
            m_grad=beta1*m_grad+(1-beta1)*log_grad
    #        Update biased second raw moment estimate
            v_grad=beta2*v_grad + (1-beta2)*np.square(log_grad)
    #        Compute bias-corrected first moment estimate
            np.divide(m_grad, 1-np.power(beta1,t), out=m_grad_new)
    #        Compute bias-corrected second raw moment estimate
            np.divide(v_grad, 1-np.power(beta2,t), out=v_grad_new)
            np.multiply(m_grad_new, alpha*-1, out=temp)
            np.divide(temp ,(np.sqrt(v_grad_new)+epsilon),out=temp)
            np.add(guess, temp, out=guess_new)

#        print("Mean abs gradient is :"+str(np.mean((np.absolute(temp)))))           
        if(np.mean((np.absolute(temp)))<tol):
            guess=guess_new
            break
        else:
            guess=guess_new
            
        if(t>max_iter):
            break;
    return guess_new
    
guess_new=adam_gd()