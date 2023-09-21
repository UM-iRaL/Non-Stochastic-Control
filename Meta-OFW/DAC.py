
"""
Disturbance Response Controller
"""
import time

import numpy as np

from headers import *
from utils import add_buffer, F_norm

FLAGS = flags.FLAGS

# DAC definition
class DAC:
    def __init__(self, par, A, B, ol, cost_func, name):
        self.d_x, self.d_u = par.d_x, par.d_u
        self.A, self.B = A, B
        self.ol = ol
        self.cost_func = cost_func
        self.name = name

        self.H, self.m = par.H, par.m
        self.LAMBDA = par.LAMBDA
        self.t = 0
        self.time = 0
        self.M = self.ol.M
        self.w_buf = []
        self.w_buf_len = self.H + self.m + 1
        self.x_pre = np.zeros((self.d_x, 1))
        self.u_pre = np.zeros((self.d_u, 1))

        # solve the ricatti equation
        self.K = par.K

    def update_noise(self, x_t):
        w_t = x_t - self.A @ self.x_pre - self.B @ self.u_pre
        # print(F_norm(w_t))
        add_buffer(self.w_buf, w_t, self.w_buf_len)
        self.x_pre = x_t

    def update_policy(self, u_t):
        self.u_pre = u_t
        if len(self.w_buf) >= self.w_buf_len:
            unary_loss, trun_loss = self.policy_loss(self.ol.M), self.trun_loss(self.ol.Ms)
            s_cost = 0 if len(self.ol.Ms) <= 1 else self.LAMBDA * F_norm(self.ol.M - self.ol.Ms[-2])
            g_t = grad(self.policy_loss)(self.M)

            # ts = time.time()
            # x = grad(self.policy_loss)(self.M)
            # te = time.time()
            # print('compute grad time: {}s'.format(te-ts))
            # exit()
            ts = time.time()
            if FLAGS.USE_GRAD:
                self.ol.opt(g_t, grad(self.policy_loss))
            else:
                self.ol.opt(g_t, self.policy_loss)
            self.M = self.ol.M
            te = time.time()
            self.time = self.time + te - ts
        else:
            unary_loss, s_cost, trun_loss = 0, 0, 0
        self.t += 1
        return unary_loss, s_cost, trun_loss

    def get_action(self, x):
        if len(self.w_buf) >= self.w_buf_len:
            return self.__action(x, self.M, 0)
        else:
            return np.zeros((self.d_u, 1))

    def policy_loss(self, M):
        time_start = time.time()
        x = np.zeros((self.d_x, 1))
        for i in range(self.H, 1, -1):
            u = self.__action(x, M, -(i+1))
            x = self.A @ x + self.B @ u + self.w_buf[-(i+1)]
        u = self.__action(x, M, -1)
        ftil_t = self.cost_func.get_cost(x, u, self.t)
        time_end = time.time()
        # print('policy loss computation:{}s'.format(time_end - time_start))
        return ftil_t

    def trun_loss(self, Ms):
        x = np.zeros((self.d_x, 1))
        for i in range(self.H, 1, -1):
            u = self.__action(x, Ms[-(i+1)], -(i+1))
            x = self.A @ x + self.B @ u + self.w_buf[-(i+1)]
        u = self.__action(x, Ms[-1], -1)
        f_t = self.cost_func.get_cost(x, u, self.t)
        return f_t

    def __action(self, x, M, t_offset):
        u = -self.K @ x
        for i in range(1, self.m + 1):
            u += M[i - 1] @ self.w_buf[t_offset - i]
        return u