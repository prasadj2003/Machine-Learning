import numpy as np

def g(z):
	return 1/(1+np.exp(-z))

def Dense(a_in, W, b):
	units = W.shape(1)
	a_out = np.zeros(units)
	for j in range(units):
		w = W[:,j]
		z = np.dot(w, a_in) + b
		a_out[j] = g(z) # g() is defined outside Dense function
	return a_out

def Sequential(x):
	a1 = Dense(x, W1, b1)
	a2 = Dense(x, W2, b2)
	a3 = Dense(x, W3, b3)
	a4 = Dense(x, W4, b4)
	f_x = a4
	return f_x