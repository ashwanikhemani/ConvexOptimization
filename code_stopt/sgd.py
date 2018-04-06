import libstopt4, sys, numpy as np


path = sys.argv[1]
lr = float(sys.argv[2])
lmda = float(sys.argv[3])
init = np.zeros((26*129+26*26), dtype=np.longdouble)

libstopt4.sgd(path, init, lr, lmda)
