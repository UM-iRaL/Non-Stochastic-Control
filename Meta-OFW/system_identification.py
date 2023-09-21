import os

import numpy as np

from headers import *
from utils import add_buffer, op_norm, F_norm

FLAGS = flags.FLAGS

def system_identification(env, T_0):
    filename = FLAGS.DATA_BASE + FLAGS.ENV + '/AB-{}.npz'.format(T_0)
    if os.path.exists(filename):
        print('Loading estimated G...')
        file = np.load(filename)
        return (file['A'], file['B'])

    max_cost = -np.inf
    x_ts, u_ts = [], []
    W = -np.inf
    if FLAGS.VERBOSE:
        print('System identifying...')

    for t in range(T_0 + 1):
        x_t = env.x
        x_ts.append(x_t)
        u_t = np.random.randint(0, 2, env.d_u).reshape((env.d_u, 1)) * 2 - 1
        u_ts.append(u_t)
        env.step(u_t)

    k = 10
    Ns = []
    for j in range(0, k + 1):
        N = np.zeros((env.d_x, env.d_u))
        for t in range(T_0 - k):
            N += x_ts[t + j + 1] @ u_ts[t].T
        N /= (T_0 - k)
        Ns.append(N)

    C_0 = Ns[0]
    for i in range(k - 1):
        C_0 = np.hstack((C_0, Ns[i + 1]))
    C_1 = Ns[1]
    for i in range(k - 1):
        C_1 = np.hstack((C_1, Ns[i + 2]))
    B = Ns[0]
    A = C_1 @ C_0.T @ np.linalg.inv(C_0 @ C_0.T)

    env.reset(0)
    for t in range(T_0 + 1):
        x_t = env.x
        x_ts.append(x_t)
        u_t = np.random.randint(0, 2, env.d_u).reshape((env.d_u, 1)) * 2 - 1
        u_ts.append(u_t)
        env.step(u_t)
        w_t = env.x - A @ x_t - B @ u_t
        W = F_norm(w_t) if F_norm(w_t) > W else W

    print('max W:{}'.format(W))
    return A, B, W
