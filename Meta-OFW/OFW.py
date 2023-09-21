from headers import *

from utils import add_buffer, F_norm

FLAGS = flags.FLAGS

class OFW:
	def __init__(self, domain, step_size):
		self.domain = domain
		self.M = self.domain.M_init
		self.step_size = step_size
		self.M_pre = self.M

	def opt(self, g=None, f=None):
		self.M_pre = self.M
		if g is None:
			g = f(self.M) if FLAGS.USE_GRAD else grad(f)(self.M)
		self.M = self.domain.projfree(self.M, self.step_size, g)
		# print(F_norm(self.M), F_norm(self.M_pre) - F_norm(self.M), F_norm(g))