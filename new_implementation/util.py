import numpy as np

def flatten(state):
     """
     能够将字典转化成数组
     """
     x = [
          state['x'], state['y'], state['z'],
          state['vx'], state['vy'], state['vz'],
          state['phi'], state['psi'], state['gamma'],
          state['vphi'], state['vpsi'], state['vgamma'],
          state['t'],
          state['theta_phi'], state['theta_psi'],
          ]
     return np.array(x, dtype=np.float32)/100.


def moving_avg(x, N=500):
     """
     train.py 里用来打印reward曲线的
     """
     if len(x) <= N:
        return []

     x_pad_left = x[0:N]
     x_pad_right = x[-N:]
     x_pad = x_pad_left[::-1] + x + x_pad_right[::-1]
     y = np.convolve(x_pad, np.ones(N) / N, mode='same')
     return y[N:-N]