# from math import gamma
import random
from tkinter.messagebox import RETRY
import numpy as np

from dynamic_helper import M1_gamma, M2_psi, M3_phi, thrust_convert_phi_psi_to_1_2, thrust_to_fx_fy_fz, omage_b_to_valtitude


state_buffer = []

dt = 0.05
g = 9.8
H = 50
I = 1/12 * H * H
target_r = 50
M = 20

world_x_min = -300  # meters
world_x_max = 300
world_y_min = -30
world_y_max = 570

f0 = 0.2 * g  # thrust
f1 = 1.0 * g
f2 = 2 * g
vphi0 = 0  # Nozzle angular velocity
vphi1 = 30 / 180 * np.pi
vphi2 = -30 / 180 * np.pi

action_table = [
    [f0, vphi0], [f0, vphi1], [f0, vphi2],
    [f1, vphi0], [f1, vphi1], [f1, vphi2],
    [f2, vphi0], [f2, vphi1], [f2, vphi2]
]

max_steps = 800
target_x, target_y, target_r = 0, H/2.0, 50

def  create_random_state():

        # predefined locations
        x_range = world_x_max - world_x_min
        y_range = world_y_max - world_y_min
        xc = (world_x_max + world_x_min) / 2.0
        yc = (world_y_max + world_y_min) / 2.0


        x = random.uniform(xc - x_range / 4.0, xc + x_range / 4.0)
        y = yc + 0.4*y_range
        if x <= 0:
            theta = -85 / 180 * np.pi
        else:
            theta = 85 / 180 * np.pi
        vy = -50

        state = {
            'x': x, 'y': y, 'vx': 0, 'vy': vy,
            'theta': theta, 'vtheta': 0,
            'phi': 0, 'f': 0,
            't': 0, 'a_': 0,
            'step_id': 0
        }

        return state

def flatten(state):
        x = [state['x'], state['y'], state['vx'], state['vy'],
             state['theta'], state['vtheta'], state['t'],
             state['phi']]
        return np.array(x, dtype=np.float32)/100.

class Rocket(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.state = create_random_state()
        self.state['already_landing'] = False
        self.state['already_crash'] = False
        return self.state



def check_landing_success(state):
    x, y = state['x'], state['y']
    vx, vy = state['vx'], state['vy']
    theta = state['theta']
    vtheta = state['vtheta']

    v = (vx**2 + vy**2)**0.5
    return True if y <= 0 + H / 2.0 and v < 15.0 and abs(x) < target_r \
                and abs(theta) < 10/180*np.pi and abs(vtheta) < 10/180*np.pi else False


def check_crash(state):
    x, y = state['x'], state['y']
    vx, vy = state['vx'], state['vy']
    theta = state['theta']
    vtheta = state['vtheta']
    v = (vx**2 + vy**2)**0.5

    crash = False
    if y >= world_y_max - H / 2.0:
        crash = True
    if y <= 0 + H / 2.0 and v >= 15.0:
        crash = True
    if y <= 0 + H / 2.0 and abs(x) >= target_r:
        crash = True
    if y <= 0 + H / 2.0 and abs(theta) >= 10/180*np.pi:
        crash = True
    if y <= 0 + H / 2.0 and abs(vtheta) >= 10/180*np.pi:
        crash = True
    return crash


def calculate_reward(state):
        x_range = world_x_max - world_x_min
        y_range = world_y_max - world_y_min

        # dist between agent and target point
        dist_x = abs(state['x'] - target_x)
        dist_y = abs(state['y'] - target_y)
        dist_norm = dist_x / x_range + dist_y / y_range

        dist_reward = 0.1*(1.0 - dist_norm)

        if abs(state['theta']) <= np.pi / 6.0:
            pose_reward = 0.1
        else:
            pose_reward = abs(state['theta']) / (0.5*np.pi)
            pose_reward = 0.1 * (1.0 - pose_reward)

        reward = dist_reward + pose_reward

        v = (state['vx'] ** 2 + state['vy'] ** 2) ** 0.5
        
        if state['already_crash']:
            reward = (reward + 5*np.exp(-1*v/10.)) * (max_steps - state['step_id'])
        if state['already_landing']: 
            reward = (1.0 + 5*np.exp(-1*v/10.))*(max_steps - state['step_id'])

        return reward



def dynamic_centriod(state, action):
    # centriod
    x, y, z = state['x'], state['y'], state['z']
    vx, vy, vz = state['vx'], state['vy'], state['vz']

    F, theta_phi, theta_psi = action['F'], action['theta_phi'], action['theta_psi']

    theta_1, theta_2 = thrust_convert_phi_psi_to_1_2(theta_phi, theta_psi)
    fx, fy, fz = thrust_to_fx_fy_fz(F, theta_1, theta_2)
    f_b = [fx, fy, fz]

    # 当前姿态
    phi, psi, gamma = state['phi'], state['psi'], state['gamma']
    f_f = M3_phi(-phi, M2_psi(-psi, M1_gamma(-gamma, f_b)))
    G = [0, -g * M, 0]

    f_f_joint = f_f + G
    ax = f_f_joint[0] / M
    ay = f_f_joint[1] / M
    az = f_f_joint[2] / M

    x_new = x + vx*dt + 0.5 * ax * (dt**2)
    y_new = y + vy*dt + 0.5 * ay * (dt**2)
    z_new = z + vz*dt + 0.5 * az * (dt**2)

    vx_new = vx + ax * dt
    vy_new = vy + ay * dt
    vz_new = vz + az * dt

    return x_new, y_new, z_new, vx_new, vy_new, vz_new

    
def dynamic_altitude(state, action):
     # 当前姿态
    phi, psi, gamma = state['phi'], state['psi'], state['gamma']

    F, theta_phi, theta_psi = action['F'], action['theta_phi'], action['theta_psi']

    theta_1, theta_2 = thrust_convert_phi_psi_to_1_2(theta_phi, theta_psi)
    fx, fy, fz = thrust_to_fx_fy_fz(F, theta_1, theta_2)
    f_b = [fx, fy, fz]
    r = [0, -H / 2, 0]
    M_b = np.cross(r, f_b)

    omega_y_b = M_b[1] / (1/12 * M * (H**2)) * dt
    omega_z_b = M_b[2] / (1/12 * M * (H**2)) * dt
    omega_x_b = 0

    vphi, vpsi, vgamma = omage_b_to_valtitude(omega_x_b, omega_y_b, omega_z_b, phi, psi, gamma)
    phi_new = phi + vphi * dt
    psi_new = psi + vpsi * dt
    gamma_new = gamma + vgamma * dt

    return phi_new, psi_new, gamma_new, vphi, vpsi, vgamma


def dynamic_thrust(state, action):
    pass


def dynamic_step(state, action):
    already_landing = state['already_landing']
    if already_landing:
        vx, vy, vz, ax, ay, ax = 0, 0, 0, 0, 0, 0
        F, theta_phi, theta_psi = 0, 0, 0
        
    step_id = state['step_id'] + 1
    x_new, y_new, z_new, vx_new, vy_new, vz_new \
     = dynamic_centriod(state, action)
    phi_new, psi_new, gamma_new, vphi, vpsi, vgamma \
     = dynamic_altitude(state, action)
    theta_phi, theta_psi = dynamic_thrust(state, action)

    

def step(state, action):
    

    # input last state
    x, y, vx, vy = state['x'], state['y'], state['vx'], state['vy'] # 质心运动

    # 绕质心运动
    theta, vtheta, \
    = state['theta'], state['vtheta']

    phi = state['phi']
    # input action
    f, vphi = action_table[action]

    ft, fr = -f*np.sin(phi), f*np.cos(phi)
    fx = ft*np.cos(theta) - fr*np.sin(theta)
    fy = ft*np.sin(theta) + fr*np.cos(theta)

    rho = 1 / (125/(g/2.0))**0.5  # suppose after 125 m free fall, then air resistance = mg
    ax, ay = fx-rho*vx, fy-g-rho*vy
    atheta = ft*H/2 / I



    #### update agent
    # 是否落地了
    already_landing = state['already_landing']
    # check if landed first
    if already_landing:
        vx, vy, ax, ay, theta, vtheta, atheta = 0, 0, 0, 0, 0, 0, 0
        phi, f = 0, 0
        action = 0

    step_id = state['step_id'] + 1
    x_new = x + vx*dt + 0.5 * ax * (dt**2)
    y_new = y + vy*dt + 0.5 * ay * (dt**2)
    vx_new, vy_new = vx + ax * dt, vy + ay * dt
    theta_new = theta + vtheta*dt + 0.5 * atheta * (dt**2)
    vtheta_new = vtheta + atheta * dt
    phi = phi + dt*vphi

    phi = max(phi, -20/180*3.1415926)
    phi = min(phi, 20/180*3.1415926)                


    already_landing = check_landing_success(state)
    already_crash = check_crash(state)

   

    if already_crash or already_landing:
        done = True
    else:
        done = False

    state = {
        'x': x_new, 'y': y_new, 'vx': vx_new, 'vy': vy_new,
        'theta': theta_new, 'vtheta': vtheta_new,
        'phi': phi, 'f': f,
        'step_id': step_id,
        't': step_id * dt,
        'already_landing': already_landing,
        'already_crash': already_crash
    }
    reward = calculate_reward(state)
    state_buffer.append(state)

    return state, reward, done, None


def test_dynamic():
    state = {
        'x': 100,
        'y': 0,
        'z': 0,
        'vx': 0,
        'vy': -10,
        'vz': 0,
        'phi': 0,
        'psi': 0,
        'gamma': 0,
        'vphi': 0,
        'vpsi': 0,
        'vgamma': 0
    }
    action = {
        'F': 0,
        'theta_phi' : 0,
        'theta_psi' : 0
    }
    print(dynamic_centriod(state, action))
    print(dynamic_altitude(state, action))


if __name__ == '__main__':
    test_dynamic()