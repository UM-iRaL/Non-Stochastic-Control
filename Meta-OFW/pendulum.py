"""
Non-PyBullet implementation of Pendulum
"""
from headers import *
from utils import op_norm_normalize, op_norm, add_buffer

FLAGS = flags.FLAGS

def angle_normalize(x):
    x = np.where(x > np.pi, x - 2*np.pi, x)
    x = np.where(x < -np.pi, x + 2*np.pi, x)
    return x

class Pendulum:
    def __init__(self):
        self.t = 0
        self.max_speed = FLAGS.MAX_SPEED
        self.max_torque = FLAGS.MAX_TORQUE
        self.dt = .05
        self.g, self.m, self.l = 10, 1, 1
        self.d_x, self.d_u = 2, 1

        self.angle_normalize = angle_normalize

        # generate approximation A, B
        # self.A_true, self.B_true = np.asarray([[1, self.dt], [(-3. * self.g * self.dt) / (2.*self.l), 1]]), np.asarray([[0], [3./(self.m*self.l**2) * self.dt]])
        # self.op_norm_A, self.op_norm_B = op_norm(self.A_true), op_norm(self.B_true)

        self.x = np.zeros((self.d_x, 1))

    def _dynamics(self, x, u):
        th, th_dot = x
        u = np.clip(u, -self.max_torque, self.max_torque)
        # newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*u) * dt
        th_dot_dot = (-3. * self.g) / (2. * self.l) * np.sin(th + np.pi) + 3. / (self.m * self.l ** 2) * u[0]
        new_th = self.angle_normalize(th + th_dot * self.dt)
        new_th_dot = th_dot + th_dot_dot * self.dt
        new_th_dot = np.clip(new_th_dot, -self.max_speed, self.max_speed)
        '''
        newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2) * u
        newth = self.angle_normalize(th + newthdot*dt)
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)'''
        x = np.array([new_th, new_th_dot])
        # print(x)
        return x

    def step(self, u):
        if not -self.max_torque <= u <= self.max_torque:
            print('Action out of bound')
            exit()
        self.t += 1
        self.x = self._dynamics(self.x, u)

    def reset(self, reset_t):
        self.t = reset_t
        self.x = np.zeros((self.d_x, 1))