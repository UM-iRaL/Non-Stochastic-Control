from headers import *

from utils import F_norm, op_norm
FLAGS = flags.FLAGS

class Domain:
	def __init__(self, par, env):
		self.par = par
		self.m = par.m
		self.d_u = env.d_u
		self.d_x = env.d_x
		self.Kdim = [env.d_u, env.d_x]
		self.K_init = np.zeros((self.d_u, self.d_x)) # par.K
		self.radius = par.R
		self.dim_vec = env.d_u * env.d_x
		self.W = par.W
		self.kappa = par.kappa
		self.gamma = par.gamma
		self.kappa = par.kappa
		self.kappa_B = par.kappa_B
		self.env = env
		self.A = env.A
		self.B = env.B
		self.Lx = env.Lx
		self.lx = env.lx
		self.Lu = env.Lu
		self.lu = env.lu
		self.xs, self.K_updates = [], []
		self.T = par.T
		self.t = 0	
		self.filename = FLAGS.RESULT_BASE + FLAGS.ENV  +'/{}-OPT.mat'.format(FLAGS.NOISE)

	def proj(self, K_update, x_t):
		# # for LTV systems
		# self.t += 1
		# self.A, self.B = self.env.ss(self.t)
		start = time.time()
		K = cp.Variable((self.d_u, self.d_x))
		constraints = []
		constraints.append(cp.norm(K + self.par.K, 2) <= self.kappa)
		constraints.append(cp.norm(self.A - self.B @ K - self.B @ self.par.K, 2) <= 1-self.gamma)		        
		constraints.append(-self.Lx @ self.B @ (K + self.par.K) @ x_t <= self.lx - self.Lx @ self.A @ x_t - np.linalg.norm(self.Lx) * self.W)
		constraints.append(-self.Lu @ (K + self.par.K) @ x_t <= self.lu)
		prob = cp.Problem(cp.Minimize(cp.norm(K_update - K, 2)), constraints)
		prob.solve()
		end = time.time()
		if FLAGS.VERBOSE:
			print("Computation time of projection: {0:.9f}".format(end - start))
			print("Control gain: ", K.value)

		if FLAGS.ENV == 'HVAC':
			self.t += 1	
			self.xs.append(x_t)
			self.K_updates.append(K_update)
			if self.t == self.T:
				scipy.io.savemat(self.filename, {"xs": self.xs, "K_updates": self.K_updates, "W": self.W,
				     							"Lx": self.Lx, "lx": self.lx, "Lu": self.Lu, "lu": self.lu,
												"kappa": self.kappa, "gamma": self.gamma, 
												"K_stab":self.par.K, "Lx_norm": np.linalg.norm(self.Lx)}) 


		return K.value
