import numpy as np

from headers import *
from utils import F_norm

FLAGS = flags.FLAGS

class Noise:
    def __init__(self, d_x, T, W=1):
        self.d_x, self.T, self.W = d_x, T, W
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
            print('Generating noises...')

            if FLAGS.ENV == 'HVAC':
                if FLAGS.NOISE == 'Gaussian':
                    self.w = np.random.normal(scale=0.5, size=(self.T + 1, self.d_x))
                elif FLAGS.NOISE == 'Uniform':
                    self.w = np.random.uniform(low=-1.2, high=1.2, size=(self.T + 1, self.d_x))
                elif FLAGS.NOISE == 'Gamma':
                    self.w = np.random.gamma(shape=4, scale=9, size=(self.T + 1, self.d_x))
                elif FLAGS.NOISE == 'Beta':
                    self.w = np.random.beta(a=4, b=9, size=(self.T + 1, self.d_x))
                elif FLAGS.NOISE == 'Exponential':
                    self.w = np.random.exponential(scale=10, size=(self.T + 1, self.d_x))
                elif FLAGS.NOISE == 'Weibull':
                    self.w = np.random.weibull(a=4, size=(self.T + 1, self.d_x))
                else:
                    print('Undefined Noise Type!')
                    exit()
                    
                for i in range(self.w.shape[0]):
                    self.w[i] = self.w[i] / max(self.w) * self.W

                filename = FLAGS.RESULT_BASE + FLAGS.ENV + '/' + '/{}-Noise.mat'.format(FLAGS.NOISE)
                scipy.io.savemat(filename, {"w": self.w})
                    
                
            elif FLAGS.ENV == 'lds':
                if FLAGS.NOISE == 'Gaussian':
                    self.w = np.random.normal(scale=0.1, size=(self.T + 1, self.d_x))
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
                else:
                    print('Undefined Noise Type!')
                    exit()
             
                for i in range(self.w.shape[0]):
                    self.w[i] = self.w[i] / F_norm(self.w[i]) * self.W

            np.savez(self.filename, w=self.w)

    def next(self):
        self.t += 1
        return self.w[self.t].reshape((self.d_x, 1))

    def reset(self, reset_t):
        self.t = reset_t - 1