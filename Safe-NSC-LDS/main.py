import copy
import os
import time

import numpy as np

from headers import *

from LinearFeedback import LinearFeedback
from domain import Domain
from OCO import OCO
from parameters import Parameters
from utils import discretize, op_norm, normalize, add_buffer
from repeated_run import Repeat_Run
from noise import Noise
from cost_function import LDS_Cost, HVAC_Cost
from lds import LDS
from HVAC import HVAC
from plot import Plot

for name in list(flags.FLAGS):
      delattr(flags.FLAGS,name)
      
FLAGS = flags.FLAGS
flags.DEFINE_string('ENV', 'HVAC', 'HVAC')
flags.DEFINE_string('DATA_BASE', './data/', 'directory of intermediate data')
flags.DEFINE_string('RESULT_BASE', './results/', 'directory of results')
flags.DEFINE_string('NOISE', 'Weibull', 'Gaussian/Uniform/Gamma/Beta/Exponential/Weibull')  
flags.DEFINE_string('COST', 'quad', 'slow/abrupt/quad')  # if 'HVAC', choose 'quad'
flags.DEFINE_string('PLOT', 'loss', 'loss/time')

flags.DEFINE_boolean('SAVE_PLOT_DATA', True, 'whether to save the data used for plotting')
flags.DEFINE_boolean('VERBOSE', False, 'whether to output intermediate results')
flags.DEFINE_boolean('CUMSUM', True, 'cumulative cost or average cost')
flags.DEFINE_boolean('DEBUG', False, 'debug mode')
flags.DEFINE_boolean('SELF_CONFIDENCE_TUNING', False, 'self-confidence tuning')
flags.DEFINE_boolean('USE_GRAD', True, 'self-confidence tuning')

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

if 'HVAC' in FLAGS.ENV:
    plot = Plot(xlabel='Iteration')
elif 'lds' in FLAGS.ENV:
    plot = Plot(xlabel='Iteration')


def lds(step_pool_scale=1, lr_meta_scale=1):
    T = 1000
    d_x, d_u = 2, 1
    W = 1
    kappa = 1
    gamma = 0.1
    D = W/gamma
    noise = Noise(d_x, T, W)
    env = LDS(T=T, d_x=d_x, d_u=d_u, noise=noise)
    cost_func = LDS_Cost(T=T, d_x=d_x, d_u=d_u)

    par = Parameters(D=D, W=W, cost_func=cost_func, T=T, kappa=kappa, gamma=gamma, step_pool_scale=step_pool_scale, lr_meta_scale=lr_meta_scale)
    par.init(env)

    params  = {'par':par, 'env':env, 'T':T, 'cost_func':cost_func, 'AB':(env.A, env.B), 'T_0':0}
    return params

def hvac(step_pool_scale=1, lr_meta_scale=1):
    T = 200
    d_x, d_u = 1, 1
    W = 1
    kappa = 5
    gamma = 0.7  # 0.5 for uniform distribution
    D = W/gamma
    noise = Noise(d_x, T, W)
    env = HVAC(T=T, d_x=d_x, d_u=d_u, noise=noise)
    cost_func = HVAC_Cost(T=T, d_x=d_x, d_u=d_u)

    par = Parameters(D=D, W=W, cost_func=cost_func, T=T, kappa=kappa, gamma=gamma, step_pool_scale=step_pool_scale, lr_meta_scale=lr_meta_scale)
    par.init(env)

    params  = {'par':par, 'env':env, 'T':T, 'cost_func':cost_func, 'AB':(env.A, env.B), 'T_0':0}
    return params

def main(seed=0, step_pool_scale=1, lr_meta_scale=1):
    if FLAGS.ENV == 'lds':
        params = lds(step_pool_scale, lr_meta_scale)
    elif FLAGS.ENV == 'HVAC':
        params = hvac(step_pool_scale, lr_meta_scale)

    par, env, T, cost_func, T_0 = params['par'], params['env'], params['T'], params['cost_func'], params['T_0']

    plot.start = T_0
    plot.par = {"step_pool_scale": step_pool_scale, "lr_meta_scale":lr_meta_scale, 'seed':seed}

    # methods = ['Safe-OGD', 'Safe-Ader', 'Safe-OGD-L']
    methods = ['Safe-OGD', 'Safe-Ader']
    # methods = ['Safe-Ader']
    # methods = ['Safe-OGD-L']
    label_list = methods

    par_ogd, par_ader = [copy.deepcopy(par) for _ in range(2)]
    ctrls = []

    if 'Safe-OGD' in methods:
        par_ogd = copy.deepcopy(par)
        par_ogd.init(env)
        par_ogd.step_pool = [par_ogd.min_step]
        par_ogd.N = 1
        domains_ogd = [Domain(par=par_ogd, env=env) for _ in range(FLAGS.REPEAT)]
        ogds = [LinearFeedback(par_ogd, env, OCO(domains_ogd[i], par_ogd), cost_func, 'Safe-OGD') for i in range(FLAGS.REPEAT)]
        ctrls.extend(ogds)

    # Safe-Ader use multiple Safe-OGD as base-learners and Hedge as meta learner, to avoid the need of step size tuning. 
    # The idea is based on: 
    # Adaptive Online Learning in Dynamic Environments (https://arxiv.org/abs/1810.10815)
    # Non-stationary Online Learning with Memory and Non-stochastic Control (https://arxiv.org/abs/2102.03758)
    if 'Safe-Ader' in methods:
        par_ader = copy.deepcopy(par)
        par_ader.init(env)
        domains_ader = [Domain(par=par_ader, env=env) for _ in range(FLAGS.REPEAT)]
        aders = [LinearFeedback(par_ader, env, OCO(domains_ader[i], par_ader), cost_func, 'Safe-Ader') for i in range(FLAGS.REPEAT)]
        ctrls.extend(aders)

    if 'Safe-OGD-L' in methods:
        par_ogd_L = copy.deepcopy(par)
        par_ogd_L.init(env)
        par_ogd_L.step_pool = [par_ogd.min_step*8]  # [par_ogd.min_step*5]
        par_ogd_L.N = 1
        domains_ogd_L = [Domain(par=par_ogd_L, env=env) for _ in range(FLAGS.REPEAT)]
        ogds_L = [LinearFeedback(par_ogd_L, env, OCO(domains_ogd_L[i], par_ogd_L), cost_func, 'Safe-OGD-L') for i in range(FLAGS.REPEAT)]
        ctrls.extend(ogds_L)

    if FLAGS.VERBOSE:
        print('Running simulation...')
    
    time_start = time.time()

    RR = Repeat_Run(env, cost_func, ctrls, T_0, T)

    time_end = time.time()
    print('Simulation time:{}s'.format(time_end - time_start))

    loss_mean_stds, state_mean_stds, input_mean_stds = RR.loss_mean_stds, RR.state_mean_stds, RR.input_mean_stds
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
        print(time_mean_stds)
        np.savez(filename, time=np.asarray(time_mean_stds))

if __name__=="__main__":
    for i in range(1):
        # time_start = time.time()
        np.random.seed(i)
        print('seed:{}'.format(i))
        main(seed=949, step_pool_scale=1, lr_meta_scale=1)
        # time_end = time.time()
        # print('run time:{}s'.format(time_end - time_start))

# # grid search
# if __name__=="__main__":
#     for i in range(1):
#         time_start = time.time()
#         np.random.seed(i)
#         print('seed:{}'.format(i))
#         step_pool_scales = [0.1, 0.5, 1, 5, 10]
#         lr_meta_scales = [0.1, 1, 10, 1e2, 1e3]
#         for step_pool_scale in step_pool_scales:
#             for lr_meta_scale in lr_meta_scales:
#                 print('step_scale:{}, lr_meta_scale:{}'.format(step_pool_scale, lr_meta_scale))
#                 main(seed=0, step_pool_scale=step_pool_scale, lr_meta_scale=lr_meta_scale)
#         time_end = time.time()
#         print('run time:{}s'.format(time_end - time_start))