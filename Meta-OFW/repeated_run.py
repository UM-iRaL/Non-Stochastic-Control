import copy
import time

import numpy as np

from headers import *

from utils import expand_list

FLAGS = flags.FLAGS

class Repeat_Run:
	def __init__(self, env, cost_func, ctls_list, T_0, T):
		self.ctls_list = ctls_list
		self.env = env
		self.T_0, self.T = T_0, T
		self.repeat = FLAGS.REPEAT
		self.cost_func = cost_func
		self.loss_mean_stds, self.unary_mean_stds, self.scost_mean_stds = [], [], []
		self.time_mean_stds = []
		self.run()

	def opt(self, ctrl):
		# print(ctrl.name)
		time_start = time.time()
		env = copy.deepcopy(self.env)
		loss, unary, s_cost = [], [], []
		for t in range(self.T_0, self.T):
			# print(t)
			x_t = env.x
			ctrl.update_noise(x_t)
			u_t = ctrl.get_action(x_t)
			if FLAGS.ENV == 'pendulum':
				u_t = np.clip(u_t, -FLAGS.MAX_TORQUE, FLAGS.MAX_TORQUE)
			env.step(u_t)
			c_t = self.cost_func.get_cost(x_t, u_t, t)
			unary_t, s_cost_t, trun_t = ctrl.update_policy(u_t)
			unary.append(unary_t)
			s_cost.append(s_cost_t)
			loss.append(c_t)
			# loss.append(trun_t)
			# print('switching cost:{}'.format(s_cost_t))
			# print('control cost:{}, OCO-M loss:{}, unary loss:{}, truncation error:{}'.format(c_t, trun_t, unary_t, c_t - trun_t))
		time_end = time.time()
		run_time = time_end - time_start
		return (loss, unary, s_cost, ctrl.name, ctrl.time)

	def run(self):
		def __loss(data_ctrls, lists):
			for data_ctrl in data_ctrls:
				data_ctrl = [np.cumsum(loss) for loss in data_ctrl]
				data_zip = [[data_ctrl[i][t] for i in range(self.repeat)] for t in range(len(data_ctrl[0]))]
				mean = np.asarray([np.mean(l) for l in data_zip])
				std = np.asarray([np.std(l) for l in data_zip])
				lists.append([mean, std])
		def __time(time_ctrls, lists):
			for time_ctrl in time_ctrls:
				lists.append([np.mean(time_ctrl), np.std(time_ctrl)])

# 		pool_run = multiprocessing.Pool()
# 		results = pool_run.map(self.opt, self.ctls_list)

		results = []
		for ctrl in self.ctls_list:
 			results.append(self.opt(ctrl))

		loss = [x[0] for x in results]
		unary = [x[1] for x in results]
		s_cost = [x[2] for x in results]
		name = [x[3] for x in results]
		time = [x[4] for x in results]
		print("Computation time in RR:", time)
		cum_loss = []
		for i in range(len(loss)):
			cum_loss.append(sum(loss[i]))
		filename = FLAGS.DATA_BASE + FLAGS.ENV + '/' + FLAGS.NOISE + '/Time-Loss-{}.npz'.format(FLAGS.COST)
		np.savez(filename, time=time, cum_loss=cum_loss)

		loss_ctrls = [loss[i:i+self.repeat] for i in range(0, len(loss), self.repeat)]
		unary_ctrls = [unary[i:i + self.repeat] for i in range(0, len(unary), self.repeat)]
		s_cost_ctrls = [s_cost[i:i + self.repeat] for i in range(0, len(s_cost), self.repeat)]
		time_ctrls = [time[i:i + self.repeat] for i in range(0, len(time), self.repeat)]

		__loss(loss_ctrls, self.loss_mean_stds)
		__loss(unary_ctrls, self.unary_mean_stds)
		__loss(s_cost_ctrls, self.scost_mean_stds)
		__time(time_ctrls, self.time_mean_stds)
