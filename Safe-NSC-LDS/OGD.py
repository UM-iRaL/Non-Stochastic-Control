from headers import *

from utils import add_buffer, F_norm

FLAGS = flags.FLAGS

class OGD:
	def __init__(self, domain, step_size):
		self.domain = domain
		self.K = self.domain.K_init
		self.step_size = step_size
		self.K_pre = self.K

	def opt(self, x_t, g=None, f=None):
		self.K_pre = self.K
		if g is None:
			g = f(self.M) if FLAGS.USE_GRAD else grad(f)(self.M)
		self.K = self.domain.proj(self.K - self.step_size * g, x_t)
