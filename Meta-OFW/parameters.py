from headers import *

from utils import discretize, op_norm

FLAGS = flags.FLAGS

class Parameters:
	def __init__(self, D=None, W=None, cost_func=None, T=None, H=None, m=None, L_c=None,
				 step_pool_scale=1, lr_meta_scale=1, use_reg=1, linearization=True, projfree=False):
		self.D, self.W, self.cost_func, self.T, self.H, self.L_c, self.m = D, W, cost_func, T, H, L_c, m
		self.step_pool_scale, self.lr_meta_scale = step_pool_scale, lr_meta_scale
		self.use_reg = use_reg
		self.linearization = linearization
		self.R = D / 2
		self.projfree = projfree

	def init(self, env):
		self.d_x, self.d_u = env.d_x, env.d_u
		dim = self.d_x if self.d_x <= self.d_u else self.d_u
		kappa_A, kappa_B = op_norm(env.A), op_norm(env.B)
		if FLAGS.VERBOSE:
			print('m:{}, H:{}'.format(self.m, self.H))

		G_c = 2
		gamma = 0.5
		# kappa = 1
		k_filename = FLAGS.DATA_BASE + FLAGS.ENV + '/' + FLAGS.NOISE + '/K.npz'
		if not os.path.exists(k_filename):
			print('generating K...')
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
		else:
			print('loading K...')
			file = np.load(k_filename)
			self.K = file['K']

		self.kappa = 0.1 if op_norm(self.K) < 0.1 else op_norm(self.K)
		
		self.gamma = gamma
		self.kappa_B = kappa_B

		if FLAGS.ENV == 'lds':
			D = self.W * self.kappa ** 3 * (1 + self.H * kappa_B**2 * self.kappa**6) / (gamma * (1 - self.kappa**2 * (1 - gamma)**(self.H + 1))) + self.W / gamma * kappa_B * self.kappa**3
			self.W += (env.var_AB * D)

		L_f = 3 * np.sqrt(self.m) * G_c * self.D * self.W * kappa_B * self.kappa**3
		G_f = 3 * self.m * dim**2 * G_c * self.W * kappa_B * self.kappa**3 / gamma
		D_f = 2 * np.sqrt(dim) * kappa_B * self.kappa**3 / gamma
		R_f = D_f / 2
		self.LAMBDA = (self.H + 2)**2 * L_f

		self.min_step = np.sqrt(6 * D ** 2 / (self.LAMBDA * D_f * self.T)) 
		self.max_step = min(1, np.sqrt((6 * D ** 2 + 2 * D ** 2 * self.T) / (self.LAMBDA * D_f * self.T))) 
		
		self.step_pool = discretize(self.min_step, self.max_step, 2)
		self.N = len(self.step_pool)

		# print("Min/Max step size is {} and {}, and step pool size is {}".format(self.min_step, self.max_step, self.step_pool))

		if FLAGS.SELF_CONFIDENCE_TUNING:
			self.lr_meta = [np.sqrt(2 / ((2 * self.LAMBDA + G_f) * (self.LAMBDA + G_f) * D_f**2 * t)) * self.lr_meta_scale for t in range(1, self.T+1)]
		else:
			self.lr_meta = np.sqrt(2 / ((2 * self.LAMBDA + G_f) * (self.LAMBDA + G_f) * D_f ** 2 * self.T)) * self.lr_meta_scale


		if FLAGS.VERBOSE:
			print('min_step:{}, max_step:{}'.format(self.min_step, self.max_step))
			print('number of base learners:{}'.format(len(self.step_pool)))
			if FLAGS.SELF_CONFIDENCE_TUNING:
				print('lr_meta_max:{}, lr_meta_min:{}'.format(self.lr_meta[0], self.lr_meta[-1]))
			else:
				print('lr_meta:{}'.format(self.lr_meta))
			print('lambda:{}, lambda/G:{}'.format(self.LAMBDA, self.LAMBDA / G_f))
