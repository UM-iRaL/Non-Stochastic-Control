import copy
import numpy as np

from headers import *

FLAGS = flags.FLAGS

class Hedge:
	def __init__(self, par):
		self.par = par
		self.expert_num = par.N
		self.lr_meta = par.lr_meta
		self.set_prior_prob()

	def set_prior_prob(self):
		self.prior_prob = np.asarray([(self.expert_num + 1) / (self.expert_num * i * (i + 1)) for i in range(1, self.expert_num + 1)])
		# self.prior_prob = np.asarray([(self.expert_num + 1) / (self.expert_num * i * (i + 1)) for i in range(self.expert_num, 0, -1)])
		self.prob = copy.deepcopy(self.prior_prob)

	def opt(self, expert_loss, t):
		expert_loss = np.asarray(expert_loss) * 1 # attention
		prob_cp = self.prob.copy()
		if FLAGS.SELF_CONFIDENCE_TUNING:
			expert_exp_loss = np.exp(-self.lr_meta[t] * expert_loss)
		else:
			expert_exp_loss = np.exp(-self.lr_meta * expert_loss)
		# print(self.lr_meta[t], expert_loss)
		self.prob = prob_cp * expert_exp_loss / np.dot(prob_cp, expert_exp_loss)
		if (np.isnan(self.prob) == True).any():
			self.prob = copy.deepcopy(self.prior_prob)
		elif True in np.isnan(self.prob):
			for i in range(self.prob.shape[0]):
				if np.isnan(self.prob[i]):
					self.prob[i] = 0
			self.prob /= np.sum(self.prob)

		# print(self.prob)