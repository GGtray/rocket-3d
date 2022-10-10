import numpy as np

def flatten(state):
        x = [state['x'], state['y'], state['vx'], state['vy'],
             state['theta'], state['vtheta'], state['t'],
             state['phi']]
        return np.array(x, dtype=np.float32)/100.