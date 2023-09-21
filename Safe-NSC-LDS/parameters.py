from headers import *

from utils import discretize, op_norm

FLAGS = flags.FLAGS

class Parameters:
	def __init__(self, D=None, W=None, cost_func=None, T=None, kappa=None, gamma=None,
				 step_pool_scale=1, lr_meta_scale=1, linearization=True):
		self.D, self.W, self.cost_func, self.T, = D, W, cost_func, T
		self.H, self.m = 1, 1
		self.kappa = kappa
		self.gamma = gamma
		self.step_pool_scale, self.lr_meta_scale = step_pool_scale, lr_meta_scale
		self.linearization = linearization
		self.R = D / 2

	def init(self, env):
		self.d_x, self.d_u = env.d_x, env.d_u
		dim = self.d_x if self.d_x <= self.d_u else self.d_u
		self.kappa_A, self.kappa_B = op_norm(env.A), op_norm(env.B)

		k_filename = FLAGS.DATA_BASE + FLAGS.ENV + '/' + FLAGS.NOISE + '/K.npz'
		if not os.path.exists(k_filename):
			print('generating K...')
			
			if FLAGS.ENV == 'lds':
				# Q, R = self.cost_func.Rs[0], self.cost_func.Ps[0]
				# X = scipy.linalg.solve_discrete_are(env.A, env.B, Q, R)
				# K = np.linalg.inv(env.B.T @ X @ env.B + R) @ (env.B.T @ X @ env.A)
				for i in range(1000):
					K = np.random.rand(self.d_u, self.d_x)
					K = K / op_norm(K) * 0.01 * i
					if op_norm(env.A - env.B @ K) < 1:
						self.K = K
						np.savez(k_filename, K=K)
						break
			elif FLAGS.ENV == 'HVAC':
				self.K = np.ones((self.d_u, self.d_x)) * 0
				np.savez(k_filename, K=self.K)
		else:
			print('loading K...')
			file = np.load(k_filename)
			self.K = file['K']
		
		if FLAGS.ENV == 'lds':
			G_c = 2
    		# stabilizing controller
			kappa_s = 1 if op_norm(self.K) < 1 else op_norm(self.K)
			gamma_s = 1-max(np.linalg.eigvals(env.A - env.B @ self.K))
			D_s = self.W * kappa_s ** 3 * (1 + 10 * self.kappa_B**2 * kappa_s**6) / (gamma_s * (1 - kappa_s**2 * (1 - gamma_s)**(10 + 1))) + self.W / gamma_s * self.kappa_B * kappa_s**3
			self.R = (op_norm(self.K) * D_s + kappa_s ** 3 * self.kappa_B / gamma_s * self.W - op_norm(self.K) * self.D)/self.D
			self.kappa = self.R + op_norm(self.K)
			if self.kappa > 1:
				self.D = self.D*self.kappa
				# # for LTV systems
				# self.W += 2*(env.var_AB * D)
		if FLAGS.ENV == 'HVAC':
			G_c = 4

		G_f = G_c * self.D * self.d_x * self.d_u * (self.kappa_B + 1)
		D_f = 2 * np.sqrt(dim) * self.kappa
		R_f = D_f / 2

		self.min_step = np.sqrt(7 * D_f ** 2 / (2 * G_f ** 2 * self.T)) # * self.step_pool_scale
		self.max_step = D_f / G_f * np.sqrt(7 / (2* self.T) + 4)  # * self.step_pool_scale 
		
		self.step_pool = discretize(self.min_step, self.max_step, 2)
		self.N = len(self.step_pool)

		# print("Min/Max step size is {} and {}, and step pool size is {}".format(self.min_step, self.max_step, self.step_pool))

		if FLAGS.SELF_CONFIDENCE_TUNING:
			self.lr_meta = [np.sqrt(2 / (G_f**2 * D_f**2 * t)) * self.lr_meta_scale for t in range(1, self.T+1)]
		else:
			self.lr_meta = np.sqrt(2 / (G_f**2 * D_f**2 * self.T))  * self.lr_meta_scale


		if FLAGS.VERBOSE:
			print('min_step:{}, max_step:{}'.format(self.min_step, self.max_step))
			print('number of base learners:{}'.format(len(self.step_pool)))
			if FLAGS.SELF_CONFIDENCE_TUNING:
				print('lr_meta_max:{}, lr_meta_min:{}'.format(self.lr_meta[0], self.lr_meta[-1]))
			else:
				print('lr_meta:{}'.format(self.lr_meta))

		# exit()