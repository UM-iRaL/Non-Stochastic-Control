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
		self.loss_mean_stds, self.state_mean_stds, self.input_mean_stds = [], [], []
		self.time_mean_stds = []
		self.run()

	def opt(self, ctrl):
		# print(ctrl.name)
		time_start = time.time()
		env = copy.deepcopy(self.env)

		loss, state, input_ctrl = [], [], []
		for t in range(self.T_0, self.T):
			x_t = env.x
			u_t = ctrl.get_action(x_t)
			
			x_next = env.step(u_t)
			ctrl.update_noise(x_next)
					
			c_t = self.cost_func.get_cost(x_next, u_t, t)
			print("cost: ", c_t)
			ctrl.update_policy(x_next, u_t)

			loss.append(c_t)
			state.append(x_t)
			input_ctrl.append(u_t)
			# print("Time, State, and Input:",t, x_t, u_t)
		time_end = time.time()
		run_time = time_end - time_start
		return (loss, state, input_ctrl, ctrl.name, ctrl.time)

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

		pool_run = multiprocessing.Pool()
		results = pool_run.map(self.opt, self.ctls_list)

# 		results = []
# 		for ctrl in self.ctls_list:
#  			results.append(self.opt(ctrl))

		loss = [x[0] for x in results]
		state = [x[1] for x in results]
		input_ctrl = [x[2] for x in results]
		name = [x[3] for x in results]
		time = [x[4] for x in results]
		print("Computation time in RR:", time)
		cum_loss = []
		for i in range(len(loss)):
			cum_loss.append(sum(loss[i]))
		print("Loss in RR:", cum_loss)

		filename = FLAGS.RESULT_BASE + FLAGS.ENV + '/Cumulative Loss.npz'
		with open(filename, 'wb') as f:
			pickle.dump((cum_loss, loss, state, input_ctrl, time, name), f)

		loss_ctrls = [loss[i:i+self.repeat] for i in range(0, len(loss), self.repeat)]
		state_ctrls = [state[i:i + self.repeat] for i in range(0, len(state), self.repeat)]
		input_ctrls = [input_ctrl[i:i + self.repeat] for i in range(0, len(input_ctrl), self.repeat)]
		time_ctrls = [time[i:i + self.repeat] for i in range(0, len(time), self.repeat)]

		__loss(loss_ctrls, self.loss_mean_stds)
		__loss(state_ctrls, self.state_mean_stds)
		__loss(input_ctrls, self.input_mean_stds)
		__time(time_ctrls, self.time_mean_stds)