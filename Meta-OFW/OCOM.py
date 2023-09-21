import numpy as np

from headers import *

from OFW import OFW
from hedge import Hedge
from utils import F_norm, add_buffer

FLAGS = flags.FLAGS

class OCOM:
	def __init__(self, domain, par):
		self.LAMBDA = par.LAMBDA
		self.domain = domain
		self.par = par
		self.experts = []
		self.expert_num = par.N
		self.hedge = Hedge(par)
		self.t = 0
		self.projfree = par.projfree
		self.init()

	def init(self):
		self.init_expert()
		self.M = self.average_M()
		self.Ms = [self.M for _ in range(self.par.H + 1)]

	def init_expert(self):
		step_pool = self.par.step_pool
		for i in range(self.expert_num):
			self.experts.append(OFW(copy.deepcopy(self.domain), step_pool[i]))
		print("OCOM Initialization with OFW experts")


	def opt(self, g_t, f_t):
		expert_loss = []
		M_pre, Ms_pre, prob_pre = self.M, [self.experts[i].M for i in range(self.expert_num)], self.hedge.prob
		for i in range(self.expert_num):
			if self.par.linearization:
				expert_loss.append(self.compute_sur_meta(i, g=g_t))
				self.experts[i].opt(g=g_t)
			else:
				expert_loss.append(self.compute_sur_meta(i, f=f_t))
				self.experts[i].opt(f=f_t)
		self.hedge.opt(expert_loss, self.t)
		self.M = self.average_M()
		Ms = [self.experts[i].M for i in range(self.expert_num)]

		# print('F-norm of M:',F_norm(self.M))
		# print('hedge prob:', list(self.hedge.prob))
		self.__check_tight(self.M, M_pre, Ms, Ms_pre, self.hedge.prob, prob_pre)
		add_buffer(self.Ms, self.M, self.par.H + 1)
		self.t += 1

	def __check_tight(self, M, M_pre, Ms, Ms_pre, prob, prob_pre):
		LHS = F_norm(M - M_pre)
		RHS = 0
		for i in range(self.expert_num):
			RHS += (prob[i] * F_norm(Ms[i] - Ms_pre[i]) + self.par.R * abs(prob[i] - prob_pre[i]))
		x = RHS - LHS
		# print(x)

	def compute_sur_meta(self, idx, g=None, f=None):
		M, M_pre = self.experts[idx].M, self.experts[idx].M_pre
		use_reg = self.par.use_reg
		action_shift = F_norm(M - M_pre)
		switching_cost = self.LAMBDA * action_shift
		if self.par.linearization == False:
			g = f(M) if FLAGS.USE_GRAD else grad(f)(M)
		# print(g.flatten() @ x.flatten(), self.LAMBDA, action_shift, use_reg * switching_cost)
		return g.flatten() @ M.flatten() + use_reg * switching_cost

	def average_M(self):
		prob = self.hedge.prob
		M_expert = [self.experts[i].M for i in range(self.expert_num)]
		M = np.zeros(M_expert[0].shape)
		for i in range(self.expert_num):
			M += (prob[i] * M_expert[i])
		if True in np.isnan(M):
			print('average_M', prob, M_expert)
			exit()
		return M