import numpy as np

from headers import *
from utils import F_norm

FLAGS = flags.FLAGS

class Noise:
    def __init__(self, d_x, T, W, sigma):
        self.d_x, self.T, self.W, self.sigma = d_x, T, W, sigma
        self.t = -1
        self.filename = FLAGS.DATA_BASE + FLAGS.ENV + '/' + FLAGS.NOISE +'/noise-{}-{}.npz'.format(d_x, T)
        if os.path.exists(self.filename):
            if FLAGS.VERBOSE:
                print('Loading noises...')
            file = np.load(self.filename)
            self.w = file['w']
        else:
            if FLAGS.VERBOSE:
                print('Generating noises...')

            if FLAGS.NOISE == 'Gaussian':
                self.w = np.random.normal(scale=self.sigma, size=(self.T + 1, self.d_x))
            elif FLAGS.NOISE == 'Uniform':
                self.w = np.random.uniform(high=0.5, size=(self.T + 1, self.d_x))
            elif FLAGS.NOISE == 'Gamma':
                self.w = np.random.gamma(shape=4, scale=9, size=(self.T + 1, self.d_x))
            elif FLAGS.NOISE == 'Beta':
                self.w = np.random.beta(a=4, b=9, size=(self.T + 1, self.d_x))
            elif FLAGS.NOISE == 'Exponential':
                self.w = np.random.exponential(scale=10, size=(self.T + 1, self.d_x))
            elif FLAGS.NOISE == 'Weibull':
                self.w = np.random.weibull(a=4, size=(self.T + 1, self.d_x))
            # elif FLAGS.NOISE == 'Poisson':
            #     self.w = np.random.poisson(lam=4, size=(self.T + 1, self.d_x))
            # elif FLAGS.NOISE == 'Binomial':
            #     self.w = np.random.binomial(n=10, p=0.5, size=(self.T + 1, self.d_x))
            else:
                print('Undefined Noise Type!')
                exit()
             
            if FLAGS.ENV == 'pendulum':
                for i in range(self.w.shape[0]):
                    self.w[i][0] = np.clip(self.w[i][0], -np.pi, np.pi)
                    self.w[i][1] = np.clip(self.w[i][1], -FLAGS.MAX_SPEED, FLAGS.MAX_SPEED)
            elif FLAGS.ENV == 'lds':
                for i in range(self.w.shape[0]):
                    self.w[i] = self.w[i] / F_norm(self.w[i]) * self.W
            np.savez(self.filename, w=self.w)

    def next(self):
        self.t += 1
        return self.w[self.t].reshape((self.d_x, 1))

    def reset(self, reset_t):
        self.t = reset_t - 1