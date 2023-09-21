import copy
import os
import time

import numpy as np

from headers import *

from DAC import DAC
from domain import Domain
from OCOM import OCOM
from parameters import Parameters
from utils import discretize, op_norm, normalize, add_buffer
from repeated_run import Repeat_Run
from noise import Noise
from cost_function import LDS_Cost, Pendulum_Cost
from lds import LDS
from pendulum import Pendulum
from plot import Plot
from system_identification import system_identification

for name in list(flags.FLAGS):
      delattr(flags.FLAGS,name)
      
FLAGS = flags.FLAGS
flags.DEFINE_string('ENV', 'lds', 'lds/pendulum')
flags.DEFINE_string('DATA_BASE', './data/', 'directory of intermediate data')
flags.DEFINE_string('RESULT_BASE', './results/', 'directory of results')
flags.DEFINE_string('NOISE', 'Gaussian', 'Gaussian/Uniform/Gamma/Beta/Exponential/Weibull')
flags.DEFINE_string('COST', 'slow', 'slow/abrupt')
flags.DEFINE_string('PLOT', 'loss', 'loss/time')

flags.DEFINE_boolean('SAVE_PLOT_DATA', True, 'whether to save the data used for plotting')
flags.DEFINE_boolean('VERBOSE', True, 'whether to output intermediate results')
flags.DEFINE_boolean('CUMSUM', True, 'cumulative cost or average cost')
flags.DEFINE_boolean('DEBUG', False, 'debug mode')
flags.DEFINE_boolean('SELF_CONFIDENCE_TUNING', False, 'self-confidence tuning')
flags.DEFINE_boolean('USE_GRAD', True, 'self-confidence tuning')

flags.DEFINE_float('SYS_ID', 2/3, 'the portion of rounds used for system identification')
flags.DEFINE_integer('REPEAT', 1, 'repeated runs')

# create folders for intermediate data and final results
if not os.path.exists(FLAGS.DATA_BASE):
    os.makedirs(FLAGS.DATA_BASE)
if not os.path.exists(FLAGS.DATA_BASE + FLAGS.ENV):
    os.makedirs(FLAGS.DATA_BASE + FLAGS.ENV)
if not os.path.exists(FLAGS.RESULT_BASE):
    os.makedirs(FLAGS.RESULT_BASE)
if not os.path.exists(FLAGS.RESULT_BASE + FLAGS.ENV):
    os.makedirs(FLAGS.RESULT_BASE + FLAGS.ENV)
if not os.path.exists(FLAGS.DATA_BASE + FLAGS.ENV + '/' + FLAGS.NOISE):
    os.makedirs(FLAGS.DATA_BASE + FLAGS.ENV + '/' + FLAGS.NOISE)

if 'lds' in FLAGS.ENV:
    plot = Plot(xlabel='Iteration')
elif FLAGS.ENV == 'pendulum':
    flags.DEFINE_float('MAX_SPEED', 20, 'the maximum speed (x[1]))')
    flags.DEFINE_float('MAX_TORQUE', 1, 'the maximum torque (u)')
    plot = Plot(xlabel='Iteration')

def lds(step_pool_scale, lr_meta_scale, projfree=False):
    T = 10000
    D = 20
    d_x, d_u = 2, 1
    noise_sigma = 0.1
    # H = int(2 * np.log(T) - 1)
    H, m = 10, 10
    W = 1
    step_pool_scale = step_pool_scale
    lr_meta_scale = lr_meta_scale

    noise = Noise(d_x, T, W, noise_sigma)
    env = LDS(T=T, d_x=d_x, d_u=d_u, noise=noise)
    cost_func = LDS_Cost(T=T, d_x=d_x, d_u=d_u)

    # A, B, W = system_identification(copy.deepcopy(env), int(T ** FLAGS.SYS_ID) * 1)

    par = Parameters(D=D, W=W, H=H, m=m, cost_func=cost_func, T=T, step_pool_scale=step_pool_scale, lr_meta_scale=lr_meta_scale, projfree=projfree)
    par.init(env)

    # print('(A-BK)^i:', [op_norm(np.linalg.matrix_power(env.A - env.B @ par.K, i)) for i in range(H)])

    params  = {'par':par, 'env':env, 'T':T, 'cost_func':cost_func, 'AB':(env.A, env.B), 'T_0':0}
    return params

def pendulum(step_pool_scale, lr_meta_scale):
    T = 10000
    L_c = np.max([2 / (3 * np.pi ** 2), 2 / (3 * FLAGS.MAX_SPEED ** 2),  2 / (3 * FLAGS.MAX_TORQUE ** 2)])
    D = 20
    d_x, d_u = 2, 1
    H, m = 5, 5

    env = Pendulum()

    cost_func = Pendulum_Cost(T)
    T_0 = int(T ** FLAGS.SYS_ID) * 1
    env.A, env.B, W = system_identification(copy.deepcopy(env), T_0)
    par = Parameters(D=D, W=W, H=H, m=m, T=T, step_pool_scale=step_pool_scale, lr_meta_scale=lr_meta_scale)
    par.init(env)

    # print('(A-BK)^i:', [op_norm(np.linalg.matrix_power(env.A - env.B @ par.K, i)) for i in range(10)])

    params = {'par':par, 'env':env, 'T_0':T_0, 'T':T, 'cost_func':cost_func, 'AB':(env.A, env.B)}
    return params

def main(seed=0, step_pool_scale=1, lr_meta_scale=1):
    if FLAGS.ENV == 'lds':
        params = lds(step_pool_scale, lr_meta_scale)
    elif FLAGS.ENV == 'pendulum':
        params = pendulum(step_pool_scale, lr_meta_scale)
    par, env, T, cost_func, T_0 = params['par'], params['env'], params['T'], params['cost_func'], params['T_0']

    plot.start = T_0
    plot.par = {"step_pool_scale": step_pool_scale, "lr_meta_scale":lr_meta_scale, 'seed':seed}

    methods = ['Meta-OFW']
    label_list = methods
 
    # domains = [Domain(dim=[par.m, env.d_u, env.d_x], radius=par.R) for _ in range(FLAGS.REPEAT)]
    ctrls = []

    if 'Meta-OFW' in methods:
        if FLAGS.ENV == 'lds':
            params_ofw = lds(step_pool_scale, lr_meta_scale, projfree=True)
            par_ofw, env_ofw, _, _, _ = params_ofw['par'], params_ofw['env'], params_ofw['T'], params_ofw['cost_func'], params_ofw['T_0']
            domains_ofw = [Domain(par=par_ofw, env=env_ofw) for _ in range(FLAGS.REPEAT)]
        metaofw = [DAC(par_ofw, env_ofw.A, env_ofw.B, OCOM(domains_ofw[i], par_ofw), cost_func, 'META-OFW') for i in range(FLAGS.REPEAT)]
        ctrls.extend(metaofw)
    if FLAGS.VERBOSE:
        print('Running simulation...')
    
    time_start = time.time()

    RR = Repeat_Run(env, cost_func, ctrls, T_0, T)

    time_end = time.time()
    print('Simulation time:{}s'.format(time_end - time_start))

    loss_mean_stds, unary_mean_stds, scost_mean_stds = RR.loss_mean_stds, RR.unary_mean_stds, RR.scost_mean_stds
    time_mean_stds = RR.time_mean_stds

    # plot
    if  FLAGS.VERBOSE:
        print('plotting...')
    if FLAGS.PLOT == 'loss':
        plot.plot_loss(loss_mean_stds, label_list, 'Cumulative Loss')
        # plot.plot_loss(unary_mean_stds, label_list, 'Unary Loss')
        # plot.plot_loss(scost_mean_stds, label_list, 'Switching Loss')
    elif FLAGS.PLOT == 'time':
        if FLAGS.ENV == 'lds' and FLAGS.COST == 'slow':
            filename = FLAGS.RESULT_BASE + 'lds/time-slow.npz'
        elif FLAGS.ENV == 'lds' and FLAGS.COST == 'abrupt':
            filename = FLAGS.RESULT_BASE + 'lds/time-abrupt.npz'
        elif FLAGS.ENV == 'pendulum':
            filename = FLAGS.RESULT_BASE + 'pendulum/time.npz'
        print(time_mean_stds)
        np.savez(filename, time=np.asarray(time_mean_stds))

if __name__=="__main__":
    for i in range(1):
        time_start = time.time()
        np.random.seed(i)
        print('seed:{}'.format(i))
        main(seed=0, step_pool_scale=1, lr_meta_scale=1)
        time_end = time.time()
        # print('run time:{}s'.format(time_end - time_start))
