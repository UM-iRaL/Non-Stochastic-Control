
"""
Linear Feedback Controller
"""
import time

import numpy as np

from headers import *
from utils import add_buffer, F_norm

FLAGS = flags.FLAGS

# Linear Feedback Policy
class LinearFeedback:
    def __init__(self, par, env, ol, cost_func, name):
        self.d_x, self.d_u = par.d_x, par.d_u
        self.env = env
        self.A, self.B = env.A, env.B
        self.Apre, self.Bpre = env.A, env.B
        self.ol = ol
        self.cost_func = cost_func
        self.name = name

        self.H, self.m = par.H, par.m
        self.t = 0
        self.time = 0
        self.K_init = self.ol.par.K # self.ol.K
        self.K = np.zeros((self.d_u, self.d_x)) # self.ol.K
        self.K_pre = np.zeros((self.d_u, self.d_x)) # self.ol.K
        self.w_buf = [np.zeros((self.d_x, 1))]
        self.w_buf_len = 1
        self.x_pre = env.x
        self.u_pre = - self.K @ env.x
        self.K = par.K

    def update_noise(self, x_t):
        # # for LTV systems
        # self.Apre, self.Bpre = self.A, self.B
        # self.A, self.B = self.env.ss(self.t)
        w_t = x_t - self.A @ self.x_pre - self.B @ self.u_pre
        self.w_buf = [w_t]
        

    def update_policy(self, x_t, u_t):
        self.u_pre = u_t
        self.K_pre = self.K
        if len(self.w_buf) >= self.w_buf_len:
            loss = self.policy_loss(self.ol.K)
            g_t = grad(self.policy_loss)(self.K)  
            print("Gradient: ", g_t)
            # ts = time.time()
            # x = grad(self.policy_loss)(self.M)
            # te = time.time()
            # print('compute grad time: {}s'.format(te-ts))
            # exit()
            ts = time.time()
            if FLAGS.USE_GRAD:
                self.ol.opt(x_t, g_t, grad(self.policy_loss))
            else:
                self.ol.opt(x_t, g_t, self.policy_loss)
            self.K = self.ol.K
            te = time.time()
            self.time = self.time + te - ts
        else:
            loss = 0
        self.x_pre = x_t
        self.t += 1

    def get_action(self, x):
        return self.__action(x)
        # if len(self.w_buf) >= self.w_buf_len:
        #     return self.__action(x)
        # else:
        #     return -self.K_init @ self.x_pre 

    def policy_loss(self, K):
        x = copy.deepcopy(self.x_pre)
        u = - (K+self.K_init) @ x
        x = self.A @ x + self.B @ u + self.w_buf[-1]
        # # for LTV systems
        # x = self.Apre @ x + self.Bpre @ u + self.w_buf[-1]
        ftil_t = self.cost_func.get_cost(x, u, self.t)

        return ftil_t

    def __action(self, x):
        u = -(self.K+self.K_init) @ x
        return u