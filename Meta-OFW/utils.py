import numpy as np

from headers import *

FLAGS = flags.FLAGS

op_norm = lambda X : np.linalg.norm(X, ord=2)
F_norm = lambda X : np.linalg.norm(X.flatten(), ord=2)
op_norm_normalize = lambda X, norm: X / op_norm(X) * norm if op_norm(X) > norm else X

def add_buffer(buffer, x, size):
	if len(buffer) >= size:
		buffer.pop(0)
	buffer.append(x)

def discretize(min_value, max_value, grid=2):
	step_pool = []
	while (min_value <= max_value):
		step_pool.append(min_value)
		min_value = min_value * grid
	step_pool.append(min_value)
	return step_pool

def angle_normalize(x):
    x = np.where(x > np.pi, x - 2*np.pi, x)
    x = np.where(x < -np.pi, x + 2*np.pi, x)
    return x

def normalize(X, max_norm, norm_type):
    norm_X = np.linalg.norm(X, ord=norm_type)
    if norm_X > max_norm:
        X = X / norm_X * max_norm
    return X

def get_G(A, B, C, H):
	G = []
	A_power = np.identity(A.shape[0])
	for i in range(H):
		add_buffer(G, C @ A_power @ B, H)
		A_power = A_power @ A
	G = np.asarray(G)
	return G

def find_closest(cand, mlist):
	r = [-1, np.inf]
	for i in range(len(mlist)):
		if abs(cand-mlist[i]) < r[1]:
			r = [i, abs(cand-mlist[i])]
	return mlist[r[0]]

def scalar_clip(x, min, max):
	if min <= x <= max:
		return x
	else:
		return np.clip(x, min, max)

def normalize_PSD(P):
	eigenvalue, _ = np.linalg.eig(P)
	print(eigenvalue)
	if np.min(eigenvalue) < 0:
		delta = - np.min(eigenvalue) + 0.001
		P = P + delta * np.identity(P.shape[0])
		eigenvalue, _ = np.linalg.eig(P)
		if np.min(eigenvalue) < 0:
			print('Not a PDS!')
			exit()
	return P

def expand_list(l):
	r = []
	for x in l:
		r.extend(x)
	return r