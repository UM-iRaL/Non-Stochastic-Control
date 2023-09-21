"""
Linear dynamical system base class
"""
import numpy as np

from headers import *
from utils import op_norm_normalize, op_norm, add_buffer, F_norm

FLAGS = flags.FLAGS

class LDS:
    """
    Description: The base, master LDS class that all other LDS subenvironments inherit. 
        Simulates a linear dynamical system with a lot of flexibility and variety in
        terms of hyperparameter choices.
    """
    def __init__(self, T=None, d_x=None, d_u=None, op_norm_A=0.9, op_norm_B=2, noise=None, var_AB=0.09):
        self.t = 0
        self.T = T
        self.d_x, self.d_u = d_x, d_u
        self.noise = noise
        self.var_AB = var_AB
        self.x = np.zeros((d_x, 1))
        self.filename = FLAGS.DATA_BASE + FLAGS.ENV + '/' + FLAGS.NOISE + '/ABs.npz'
        if os.path.exists(self.filename):
            if FLAGS.VERBOSE:
                print('Loading system dynamics...')
            file = np.load(self.filename)
            self.As, self.Bs, self.A, self.B = list(file['As']), list(file['Bs']), file['A'], file['B']
        else:
            if FLAGS.VERBOSE:
                print('Generating system dynamics...')

            if FLAGS.NOISE == 'Gaussian':
                self.A = op_norm_normalize(np.random.normal(size=(d_x, d_x)), op_norm_A)
                self.B = op_norm_normalize(np.random.normal(size=(d_x, d_u)), op_norm_B)
            elif FLAGS.NOISE == 'Uniform':
                self.A = op_norm_normalize(np.random.uniform(high=0.5, size=(d_x, d_x)), op_norm_A)
                self.B = op_norm_normalize(np.random.uniform(high=0.5, size=(d_x, d_u)), op_norm_B)
            elif FLAGS.NOISE == 'Gamma':
                self.A = op_norm_normalize(np.random.gamma(shape=4, scale=9, size=(d_x, d_x)), op_norm_A)
                self.B = op_norm_normalize(np.random.gamma(shape=4, scale=9, size=(d_x, d_u)), op_norm_B)
            elif FLAGS.NOISE == 'Beta':
                self.A = op_norm_normalize(np.random.beta(a=4, b=9, size=(d_x, d_x)), op_norm_A)
                self.B = op_norm_normalize(np.random.beta(a=4, b=9, size=(d_x, d_u)), op_norm_B)
            elif FLAGS.NOISE == 'Exponential':
                self.A = op_norm_normalize(np.random.exponential(scale=10.0, size=(d_x, d_x)), op_norm_A)
                self.B = op_norm_normalize(np.random.exponential(scale=10.0, size=(d_x, d_u)), op_norm_B)
            elif FLAGS.NOISE == 'Weibull':
                self.A = op_norm_normalize(np.random.weibull(a=4, size=(d_x, d_x)), op_norm_A)
                self.B = op_norm_normalize(np.random.weibull(a=4, size=(d_x, d_u)), op_norm_B)
            # elif FLAGS.NOISE == 'Poisson':
            #     self.A = op_norm_normalize(np.random.poisson(lam=4, size=(d_x, d_x)), op_norm_A)
            #     self.B = op_norm_normalize(np.random.poisson(lam=4, size=(d_x, d_u)), op_norm_B)
            # elif FLAGS.NOISE == 'Binomial':
            #     self.A = op_norm_normalize(np.random.binomial(n=10, p=0.5, size=(d_x, d_x)), op_norm_A)
            #     self.B = op_norm_normalize(np.random.binomial(n=10, p=0.5, size=(d_x, d_u)), op_norm_B)
            else:
                print('Undefined Noise Type!')
                exit()

            self.As, self.Bs = [], []
            for i in range(self.T+1):
                if FLAGS.NOISE == 'Gaussian':
                    drift_A = op_norm_normalize(np.random.normal(loc=0, scale=var_AB, size=(d_x, d_x)), var_AB)
                    drift_B = op_norm_normalize(np.random.normal(loc=0, scale=var_AB, size=(d_x, d_u)), var_AB)
                elif FLAGS.NOISE == 'Uniform':
                    drift_A = op_norm_normalize(np.random.uniform(high=0.25, size=(d_x, d_x)), var_AB)
                    drift_B = op_norm_normalize(np.random.uniform(high=0.25, size=(d_x, d_u)), var_AB)
                elif FLAGS.NOISE == 'Gamma':
                    drift_A = op_norm_normalize(np.random.gamma(shape=4, scale=9, size=(d_x, d_x)), var_AB)
                    drift_B = op_norm_normalize(np.random.gamma(shape=4, scale=9, size=(d_x, d_u)), var_AB)
                elif FLAGS.NOISE == 'Beta':
                    drift_A = op_norm_normalize(np.random.beta(a=4, b=9, size=(d_x, d_x)), var_AB)
                    drift_B = op_norm_normalize(np.random.beta(a=4, b=9, size=(d_x, d_u)), var_AB)
                elif FLAGS.NOISE == 'Exponential':
                    drift_A = op_norm_normalize(np.random.exponential(scale=10.0, size=(d_x, d_x)), var_AB)
                    drift_B = op_norm_normalize(np.random.exponential(scale=10.0, size=(d_x, d_u)), var_AB)
                elif FLAGS.NOISE == 'Weibull':
                    drift_A = op_norm_normalize(np.random.weibull(a=4, size=(d_x, d_x)), var_AB)
                    drift_B = op_norm_normalize(np.random.weibull(a=4, size=(d_x, d_u)), var_AB)
                # elif FLAGS.NOISE == 'Poisson':
                #     drift_A = op_norm_normalize(np.random.poisson(lam=4, size=(d_x, d_x)), var_AB)
                #     drift_B = op_norm_normalize(np.random.poisson(lam=4, size=(d_x, d_u)), var_AB)
                # elif FLAGS.NOISE == 'Binomial':
                #     drift_A = op_norm_normalize(np.random.binomial(n=10, p=0.5, size=(d_x, d_x)), var_AB)
                #     drift_B = op_norm_normalize(np.random.binomial(n=10, p=0.5, size=(d_x, d_u)), var_AB)
                else:
                    print('Undefined Noise Type!')
                    exit()

                cur_A, cur_B = self.A + drift_A, self.B + drift_B
                self.As.append(cur_A)
                self.Bs.append(cur_B)
            np.savez(self.filename, A=self.A, B=self.B, As=np.asarray(self.As), Bs=np.asarray(self.Bs))

    def step(self, u):
        w_t = self.noise.next()
        # print(F_norm(w_t))
        self.x = self.As[self.t] @ self.x + self.Bs[self.t] @ u + w_t
        self.t += 1

    def reset(self, reset_t):
        self.noise.reset(reset_t)
        self.t = reset_t
        self.x = np.zeros(self.d_x)