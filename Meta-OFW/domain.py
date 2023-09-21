from headers import *

from utils import F_norm, op_norm
FLAGS = flags.FLAGS

class Domain:
	def __init__(self, par, env):
		self.m = par.m
		self.d_u = env.d_u
		self.d_x = env.d_x
		self.dim = [par.m, env.d_u, env.d_x]
		self.radius = par.R
		self.dim_vec = par.m * env.d_u * env.d_x
		self.gamma = par.gamma
		self.kappa = par.kappa
		self.kappa_B = par.kappa_B
		self.M_init = self.init()

	def init(self):
		M_init = np.random.randn(self.dim_vec).reshape(tuple(self.dim))
		M_init = M_init / F_norm(M_init) * self.radius * 0.3
		return M_init

	def projfree(self, M, step_size, g):
		assert step_size <= 1
		M_prime = np.zeros(g.shape)
		norm_cons = self.radius/self.m # self.kappa_B * self.kappa ** 3 * (1-self.gamma) ** i
		
		# 1-D version for projection-free update
		# M_prime = copy.deepcopy(-1.0*g)
		# start = time.time()
		# for i in range(self.m):
		# 	opnorm = op_norm(M_prime[i])
		# 	M_prime[i] = M_prime[i] / opnorm * norm_cons
		# end = time.time()
		# if FLAGS.VERBOSE:
		# 	print("Computation time of preojection-free update: {0:.9f}".format(end - start))
		# return (1-step_size)*M + step_size*M_prime


		# General version for projection-free update
		for i in range(self.m):
			Mi_prime = cp.Variable((self.d_u, self.d_x))
			constraints = []
			constraints = [cp.norm(Mi_prime, 2) <= norm_cons]
			prob = cp.Problem(cp.Minimize(cp.trace(Mi_prime.T @ g[i])), constraints)
			# start = time.time()
			prob.solve()
			# end = time.time()
			# if FLAGS.VERBOSE:
			# 	print("Computation time of {}-th preojection-free update: {}".format(i, end - start))
			M_prime[i] = Mi_prime.value
		return (1-step_size)*M + step_size*M_prime