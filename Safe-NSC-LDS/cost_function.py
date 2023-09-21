import numpy as np

from headers import *
from utils import op_norm_normalize, angle_normalize, scalar_clip, op_norm, normalize_PSD

FLAGS = flags.FLAGS

class LDS_Cost:
	def __init__(self, d_x=None, d_u=None, T=None, stages=5, op_norm_R=1, op_norm_P=1):
		self.Rs, self.Ps = [], []
		filename = FLAGS.DATA_BASE + FLAGS.ENV + '/' + FLAGS.NOISE + '/R-P-{}-{}-{}-{}.npz'.format(FLAGS.COST, d_x, d_u, T)
		if not os.path.exists(filename):
			print('Generating cost functions...')
			if FLAGS.COST == 'slow':
				# cur_R = op_norm_normalize(np.random.rand(d_x, d_x), op_norm_R)
				# cur_P = op_norm_normalize(np.random.rand(d_u, d_u), op_norm_P)
				cur_R = (np.sin(1 / (20 * np.pi)) + 1) * np.identity(d_x)
				cur_P = (np.sin(1 / (10 * np.pi)) + 1) * np.identity(d_u)
				for i in range(T):
					self.Rs.append(cur_R)
					self.Ps.append(cur_P)
					cur_R = (np.sin((i + 1) / (20 * np.pi)) + 1) * np.identity(d_x)
					cur_P = (np.sin((i + 1) / (10 * np.pi)) + 1) * np.identity(d_u)
					# self.Rs.append(cur_R @ cur_R.T)
					# self.Ps.append(cur_P @ cur_P.T)
					# drift_R = np.random.normal(loc=0, scale=1, size=(d_x, d_x))
					# drift_R = drift_R / op_norm(drift_R) * var_R
					# drift_P = np.random.normal(loc=0, scale=1, size=(d_u, d_u))
					# drift_P = drift_P / op_norm(drift_P) * var_P
					# cur_R, cur_P = cur_R + drift_R, cur_P + drift_P
					# cur_R = op_norm_normalize(cur_R, op_norm_R)
					# cur_P = op_norm_normalize(cur_P, op_norm_P)
			elif FLAGS.COST == 'abrupt':
				x = np.log(2) / 2
				R = [x, 1, x, 1, x]
				P = [1, 1, x, x, 1]
				for i in range(stages):
					cur_R = R[i] * np.identity(d_x)
					cur_P = P[i] * np.identity(d_u)
					for j in range(int(T/stages)+1):
						self.Rs.append(cur_R)
						self.Ps.append(cur_P)
				self.Rs = self.Rs[:T]
				self.Ps = self.Ps[:T]
			else:
				exit()
			if len(self.Rs) != T or len(self.Ps) != T:
				print('Length error!')
				exit()
			np.savez(filename, R=np.asarray(self.Rs), P=np.asarray(self.Ps))
		else:
			print('Loading cost functions...')
			file = np.load(filename)
			self.Rs, self.Ps = list(file['R']), list(file['P'])

		# norm = set([op_norm(x) for x in self.Rs])
		# print(norm)
		# exit()

	def get_cost(self, x, u, t):
		# print(t)
		c_t = (x.T @ self.Rs[t] @ x + u.T @ self.Ps[t] @ u) * 1
		# c_t = (x.T @ x + u.T @ u) * 1
		# print(x, self.Rs[t], u, self.Ps[t])
		return c_t[0][0]


class HVAC_Cost:
	def __init__(self, T=200, d_x=1, d_u=1):
		filename = FLAGS.DATA_BASE + FLAGS.ENV + '/' + FLAGS.NOISE + '/cost.npz'
		if os.path.exists(filename):
			print('Loading cost functions...')
			file = np.load(filename)
			self.q, self.r = list(file['q']), list(file['r'])
		else:
			print('Generating cost functions...')
			self.q, self.r = [], []
			q = np.identity(d_x) # * 2 / 2
			r = np.identity(d_u) # * np.random.uniform(low=0.1, high=4) / 2
			for i in range(T):
				self.q.append(q)
				self.r.append(r)
				q = np.identity(d_x) # * 2 / 2
				r = np.identity(d_u) # * np.random.uniform(low=0.1, high=4) / 2
			if len(self.q) != T or len(self.r) != T:
				print('Length error!')
				exit()
			np.savez(filename, q=np.asarray(self.q), r=np.asarray(self.r))

	def get_cost(self, x, u, t):
		c_t = (x.T) @ self.q[t] @ (x) + (u.T) @ self.r[t] @ (u)
		return c_t[0][0]

