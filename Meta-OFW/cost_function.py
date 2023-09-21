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


class Pendulum_Cost:
	def __init__(self, T, var_p=0.1, abs_p=1):
		filename = FLAGS.DATA_BASE + FLAGS.ENV + '/cost.npz'
		if os.path.exists(filename):
			print('Loading cost functions...')
			file = np.load(filename)
			self.p1, self.p2, self.p3 = list(file['p1']), list(file['p2']), list(file['p3'])
		else:
			print('Generating cost functions...')
			self.p1, self.p2, self.p3 = [], [], []
			cur_p1 = (np.sin(1 / (10 * np.pi)) + 1) / 2
			cur_p2 = (np.sin(1 / (20 * np.pi)) + 1) / 2
			cur_p3 = (np.sin(1 / (30 * np.pi)) + 1) / 2
			# cur_p1, cur_p2, cur_p3 = np.random.rand(), np.random.rand(), np.random.rand()
			for i in range(T):
				self.p1.append(cur_p1)
				self.p2.append(cur_p2)
				self.p3.append(cur_p3)
				cur_p1 = (np.sin((i + 1) / (10 * np.pi)) + 1) / 2
				cur_p2 = (np.sin((i + 1) / (20 * np.pi)) + 1) / 2
				cur_p3 = (np.sin((i + 1) / (30 * np.pi)) + 1) / 2
				# drift_p1 = np.random.normal(loc=var_p/2, scale=1, size=1)[0]
				# drift_p1 = scalar_clip(drift_p1, 0, var_p)
				# drift_p2 = np.random.normal(loc=var_p/2, scale=1, size=1)[0]
				# drift_p2 = scalar_clip(drift_p2, 0, var_p)
				# drift_p3 = np.random.normal(loc=var_p/2, scale=1, size=1)[0]
				# drift_p3 = scalar_clip(drift_p3, 0, var_p)
				# cur_p1, cur_p2, cur_p3 = cur_p1 + drift_p1, cur_p2 + drift_p2, cur_p3 + drift_p3
				# cur_p1, cur_p2, cur_p3 = scalar_clip(cur_p1, 0, abs_p), scalar_clip(cur_p2, 0, abs_p), scalar_clip(cur_p3, 0, abs_p)
			if len(self.p1) != T or len(self.p2) != T:
				print('Length error!')
				exit()
			np.savez(filename, p1=np.asarray(self.p1), p2=np.asarray(self.p2), p3=np.asarray(self.p3))

	def get_cost(self, x, u, t):
		c_t = self.p1[t] * angle_normalize(x[0][0]) ** 2 / (3 * np.pi**2) + self.p2[t] * x[1][0] ** 2 / (3 * FLAGS.MAX_SPEED**2) + self.p3[t] * u[0][0] ** 2 / (3 * FLAGS.MAX_TORQUE**2)
		# c_t = angle_normalize(x[0][0]) ** 2 / (3 * np.pi ** 2) + x[1][0] ** 2 / (
					# 3 * FLAGS.MAX_SPEED ** 2) + u[0][0] ** 2 / (3 * FLAGS.MAX_TORQUE ** 2)
		return c_t * 1

