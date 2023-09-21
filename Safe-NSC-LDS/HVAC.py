"""
Linear dynamical system base class
"""
import numpy as np

from headers import *
from utils import op_norm_normalize, op_norm, add_buffer, F_norm

FLAGS = flags.FLAGS

class HVAC:
    """
    HVAC System 
    xdot = 1/(v*zeta) * (theta0 - x) - 1/v*u + 1/v*pi + 1/v*wt
         = -1/(v*zeta)*x - 1/v*u + 1/(v*zeta)*theta0 + 1/v*wt
    v = 100;
    zeta = 6;
    theta0 = 30;
    pi = 1.5;
    dt = 60;    
    """
    def __init__(self, T=None, d_x=None, d_u=None, op_norm_A=0.9, op_norm_B=2, noise=None, var_AB=0.09):
        self.t = 0
        self.T = T
        self.d_x, self.d_u = 1, 1
        self.noise = noise
        self.x = np.zeros((d_x, 1)) 
        self.Lx = np.array([[1],[-1]])
        self.lx = np.array([[2], [2]])
        self.Lu = np.array([[1],[-1]])
        self.lu = np.array([[2.5], [2.5]])
        self.A = np.ones((d_x, d_x)) * 0.9
        self.B = np.ones((d_x, d_u)) * 0.6
        self.K = np.ones((d_u, d_x)) * 0
        self.filename = FLAGS.DATA_BASE + FLAGS.ENV + '/' + FLAGS.NOISE + '/ABK.npz'

    def step(self, u):
        w_t = self.noise.next()
        # print(F_norm(w_t))
        # noise is self.Ed @ w_t + self.Ec 
        self.x = self.A @ self.x + self.B @ u + w_t 
        self.t += 1
        if self.x > 2 or self.x < -2:
            print("State violates safety constrainsts!")
            exit()
        if u > 2.5 or u < -2.5:
            print("Control violates safety constrainsts!")
            exit()
        return self.x
        
    # # for LTV systems
    def ss(self, t):
        return self.A, self.B

    def reset(self, reset_t):
        self.noise.reset(reset_t)
        self.t = reset_t
        self.x = np.zeros((self.d_x, 1)) 