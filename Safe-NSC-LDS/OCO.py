import numpy as np

from headers import *

from OGD import OGD
from hedge import Hedge
from utils import F_norm, add_buffer

FLAGS = flags.FLAGS

class OCO:
	def __init__(self, domain, par):
		self.domain = domain
		self.par = par
		self.experts = []
		self.expert_num = par.N
		self.hedge = Hedge(par)
		self.t = 0
		self.init()

	def init(self):
		self.init_expert()
		self.K = self.average_K()
		self.Ks = [self.K for _ in range(self.par.H + 1)]

	def init_expert(self):
		step_pool = self.par.step_pool
		for i in range(self.expert_num):
			self.experts.append(OGD(copy.deepcopy(self.domain), step_pool[i]))
		print("OCO Initialization with OGD experts")

	def opt(self, x_t, g_t, f_t):
		expert_loss = []
		# K_pre, Ks_pre, prob_pre = self.K, [self.experts[i].K for i in range(self.expert_num)], self.hedge.prob
		for i in range(self.expert_num):
			if self.par.linearization:
				expert_loss.append(self.compute_sur_meta(i, g=g_t))
				self.experts[i].opt(x_t=x_t, g=g_t)
			else:
				expert_loss.append(self.compute_sur_meta(i, f=f_t))
				self.experts[i].opt(x_t=x_t, f=f_t)
		self.hedge.opt(expert_loss, self.t)
		self.K = self.average_K()
		print("Hedge Prob: ", self.hedge.prob)
		# Ks = [self.experts[i].K for i in range(self.expert_num)]

		# print('F-norm of M:',F_norm(self.M))
		# print('hedge prob:', list(self.hedge.prob))
		# self.__check_tight(self.M, M_pre, Ms, Ms_pre, self.hedge.prob, prob_pre)
		add_buffer(self.Ks, self.K, self.par.H + 1)
		self.t += 1

	# def __check_tight(self, M, M_pre, Ms, Ms_pre, prob, prob_pre):
	# 	LHS = F_norm(M - M_pre)
	# 	RHS = 0
	# 	for i in range(self.expert_num):
	# 		RHS += (prob[i] * F_norm(Ms[i] - Ms_pre[i]) + self.par.R * abs(prob[i] - prob_pre[i]))
	# 	x = RHS - LHS
	# 	# print(x)

	def compute_sur_meta(self, idx, g=None, f=None):
		# if self.par.linearization == False:
		# 	g = f(M) if FLAGS.USE_GRAD else grad(f)(M)
		# print(g.flatten() @ x.flatten(), self.LAMBDA, action_shift, use_reg * switching_cost)
		return g.flatten() @ (self.experts[idx].K - self.K).flatten() 

	def average_K(self):
		prob = self.hedge.prob
		K_expert = [self.experts[i].K for i in range(self.expert_num)]
		K = np.zeros(K_expert[0].shape)
		for i in range(self.expert_num):
			K += (prob[i] * K_expert[i])
		if True in np.isnan(K):
			print('average_K', prob, K_expert)
			exit()
		return K