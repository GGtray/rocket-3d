# from math import gamma
import random
from tkinter.messagebox import RETRY
import numpy as np

from dynamic_helper import M1_gamma, M2_psi, M3_phi, thrust_convert_phi_psi_to_1_2, thrust_to_fx_fy_fz, omage_b_to_vattitude


state_buffer = {}

dt = 0.05
g = 9.8
H = 20
I = 1/12 * H ** 2
target_r = 50
M = 20

world_x_min = -300  # meters
world_x_max = 300
world_y_min = -30
world_y_max = 570
world_z_min = -300
world_z_max = 300

max_steps = 800
target_x, target_z, target_y, target_r = 0, 0, H/2.0, 50
def create_action_table_1d( f0=1.2*9.8*M, f1=2.0*9.8*M, f2=3*9.8*M):
    action_table = [f0, f1, f2]
    return action_table


def create_action_table(
    f0=0.5*9.8*M, f1=1.0*9.8*M, f2=2*9.8*M, 
    vtheta_phi_0=0, vtheta_phi_1=30 / 180 * np.pi, vtheta_phi_2=-30 / 180 * np.pi,
    vtheta_psi_0=0, vtheta_psi_1=30 / 180 * np.pi, vtheta_psi_2=-30 / 180 * np.pi
    ):

    action_table = [
    [f0, vtheta_phi_0, vtheta_psi_0], [f0, vtheta_phi_1, vtheta_psi_0], [f0, vtheta_phi_2, vtheta_psi_0],
    [f1, vtheta_phi_0, vtheta_psi_0], [f1, vtheta_phi_1, vtheta_psi_0], [f1, vtheta_phi_2, vtheta_psi_0],
    [f2, vtheta_phi_0, vtheta_psi_0], [f2, vtheta_phi_1, vtheta_psi_0], [f2, vtheta_phi_2, vtheta_psi_0],

    [f0, vtheta_phi_0, vtheta_psi_1], [f0, vtheta_phi_1, vtheta_psi_1], [f0, vtheta_phi_2, vtheta_psi_1],
    [f1, vtheta_phi_0, vtheta_psi_1], [f1, vtheta_phi_1, vtheta_psi_1], [f1, vtheta_phi_2, vtheta_psi_1],
    [f2, vtheta_phi_0, vtheta_psi_1], [f2, vtheta_phi_1, vtheta_psi_1], [f2, vtheta_phi_2, vtheta_psi_1],
    
    [f0, vtheta_phi_0, vtheta_psi_2], [f0, vtheta_phi_1, vtheta_psi_2], [f0, vtheta_phi_2, vtheta_psi_2],
    [f1, vtheta_phi_0, vtheta_psi_2], [f1, vtheta_phi_1, vtheta_psi_2], [f1, vtheta_phi_2, vtheta_psi_2],
    [f2, vtheta_phi_0, vtheta_psi_2], [f2, vtheta_phi_1, vtheta_psi_2], [f2, vtheta_phi_2, vtheta_psi_2],
    ] # 这个action table非常大，还有一种办法是变成一个时刻只能动一个方向

    return action_table

def create_action_table_s(
   f0=0.2*9.8*M, f1=1.0*9.8*M, f2=2*9.8*M, 
    vtheta_phi_0=0, vtheta_phi_1=30 / 180 * np.pi, vtheta_phi_2=-30 / 180 * np.pi,
    vtheta_psi_0=0, vtheta_psi_1=30 / 180 * np.pi, vtheta_psi_2=-30 / 180 * np.pi
    ):

    action_table = [
    [f0, vtheta_phi_0, vtheta_psi_0], [f0, vtheta_phi_1, vtheta_psi_0], [f0, vtheta_phi_2, vtheta_psi_0],
    [f1, vtheta_phi_0, vtheta_psi_0], [f1, vtheta_phi_1, vtheta_psi_0], [f1, vtheta_phi_2, vtheta_psi_0],
    [f2, vtheta_phi_0, vtheta_psi_0], [f2, vtheta_phi_1, vtheta_psi_0], [f2, vtheta_phi_2, vtheta_psi_0],

    [f0, vtheta_phi_0, vtheta_psi_1], [f0, vtheta_phi_1, vtheta_psi_1], [f0, vtheta_phi_2, vtheta_psi_1],
    [f1, vtheta_phi_0, vtheta_psi_1], [f1, vtheta_phi_1, vtheta_psi_1], [f1, vtheta_phi_2, vtheta_psi_1],
    [f2, vtheta_phi_0, vtheta_psi_1], [f2, vtheta_phi_1, vtheta_psi_1], [f2, vtheta_phi_2, vtheta_psi_1],
    
    [f0, vtheta_phi_0, vtheta_psi_2], [f0, vtheta_phi_1, vtheta_psi_2], [f0, vtheta_phi_2, vtheta_psi_2],
    [f1, vtheta_phi_0, vtheta_psi_2], [f1, vtheta_phi_1, vtheta_psi_2], [f1, vtheta_phi_2, vtheta_psi_2],
    [f2, vtheta_phi_0, vtheta_psi_2], [f2, vtheta_phi_1, vtheta_psi_2], [f2, vtheta_phi_2, vtheta_psi_2],
    ] # 这个action table非常大，还有一种办法是变成一个时刻只能动一个方向

    return action_table

def  create_random_start_state():

        # predefined locations
        x_range = world_x_max - world_x_min
        y_range = world_y_max - world_y_min
        z_range = world_z_max - world_z_min

        xc = (world_x_max + world_x_min) / 2.0
        zc = (world_z_max + world_z_min) / 2.0
        yc = (world_y_max + world_y_min) / 2.0


        x = random.uniform(xc - x_range / 4.0, xc + x_range / 4.0)
        z = random.uniform(zc - z_range / 4.0, zc + z_range / 4.0)
        y = yc + 0.4*y_range


        phi, psi = np.pi/2, 0
        
        vy = -50

        state = {
            'x': 0, 'y': 500, 'z': 0,
            'vx': 0, 'vy': vy, 'vz': 0,
            'phi': phi, 'psi': psi, 'gamma': 0,
            'vphi': 0, 'vpsi':0, 'vgamma': 0,
            'theta_phi': 0, 'theta_psi': 0, 'F': 0,
            't': 0, 'step_id': 0
        }

        return state

def flatten(state):
        x = [
            state['x'], state['y'], state['z'], 
            state['vx'], state['vy'], state['vz'],
            state['phi'], state['psi'], state['gamma'],
            state['vphi'], state['vpsi'], state['vgamma']
            ]
        return np.array(x, dtype=np.float32)/100.

class Rocket(object):

    def __init__(self):
        self.init()

    def init(self):
        self.state = create_random_start_state()
        self.state['already_landing'] = False
        self.state['already_crash'] = False
        return self.state




def check_landing_success(state):
    x, y, z = state['x'], state['y'], state['z']
    vx, vy, vz = state['vx'], state['vy'], state['vz']
    phi, psi, gamma = state['phi'], state['psi'], state['gamma']
    vphi, vpsi, vgamma = state['vphi'], state['vpsi'], state['vgamma']


    v = (vx**2 + vy**2 + vz**2)**0.5
    return True \
        if y <= 0 + H / 2.0 \
        and v < 15.0 and np.sqrt(x**2 + z**2) < target_r \
        and abs(phi) < 10/180*np.pi \
        and abs(psi) < 10/180*np.pi \
        and abs(gamma) < 10/180*np.pi \
        and abs(vphi)  < 10/180*np.pi \
        and abs(vpsi)  < 10/180*np.pi \
        and abs(vgamma)  < 10/180*np.pi \
        else False


def check_crash(state):
    x, y, z = state['x'], state['y'], state['z']
    vx, vy, vz = state['vx'], state['vy'], state['vz']
    phi, psi, gamma = state['phi'], state['psi'], state['gamma']
    vphi, vpsi, vgamma = state['vphi'], state['vpsi'], state['vgamma']

    v = (vx**2 + vy**2 + vz**2)**0.5

    crash = False
    if y >= world_y_max - H / 2.0:
        crash = True
    if y <= 0 + H / 2.0 and v >= 15.0:
        crash = True
    if y <= 0 + H / 2.0 and np.sqrt(x**2 + z**2) >= target_r:
        crash = True
    if y <= 0 + H / 2.0 and  ((abs(phi) >= 10/180*np.pi) or (abs(psi) >= 10/180*np.pi) or (abs(gamma) >= 10/180*np.pi)) :
        crash = True
    if y <= 0 + H / 2.0 and ((abs(vphi) >= 10/180*np.pi) or (abs(vpsi) >= 10/180*np.pi) or (abs(vgamma) >= 10/180*np.pi)) :
        crash = True
    return crash


def calculate_reward(state):
        x_range = world_x_max - world_x_min
        y_range = world_y_max - world_y_min
        z_range = world_z_max - world_z_min

        # dist between agent and target point
        dist_x = abs(state['x'] - target_x)
        dist_y = abs(state['y'] - target_y)
        dist_z = abs(state['z'] - target_z)

        loc_norm = dist_x / x_range + dist_z / z_range + dist_y / y_range

        loc_reward = 0.1*(1.0 - loc_norm)

        dist_phi = abs(state['phi'] - 90/180*np.pi)
        dist_psi = abs(state['psi'])
        dist_gamma = abs(state['gamma'])

        att_norm = dist_phi + dist_psi + dist_gamma

        if att_norm <= np.pi / 6.0:
            att_reward = 0.1
        else:
            att_reward = 0.1 * (1.0 - att_norm / (0.5 * np.pi))

        reward = loc_reward + att_reward

        v = (state['vx'] ** 2 + state['vy'] ** 2 + state['vz'] ** 2) ** 0.5
        
        if state['already_crash']:
            reward = (reward + 5*np.exp(-1*v/10.)) * (max_steps - state['step_id'])
        if state['already_landing']: 
            reward = (1.0 + 5*np.exp(-1*v/10.))*(max_steps - state['step_id'])

        return reward



def dynamic_centriod(state, action):
    # centriod
    x, y, z = state['x'], state['y'], state['z']
    vx, vy, vz = state['vx'], state['vy'], state['vz']

    # F, theta_phi, theta_psi = action
    # for 1d exp
    F = action
    theta_phi, theta_psi = 0, 0

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

    
def dynamic_attitude(state, action):
     # 当前姿态
    phi, psi, gamma = state['phi'], state['psi'], state['gamma']

    # F, theta_phi, theta_psi = action # for 1d expr
    F = action
    theta_phi = 0
    theta_psi = 0

    theta_1, theta_2 = thrust_convert_phi_psi_to_1_2(theta_phi, theta_psi)
    fx, fy, fz = thrust_to_fx_fy_fz(F, theta_1, theta_2)
    f_b = [fx, fy, fz]
    r = [0, -H / 2, 0]
    M_b = np.cross(r, f_b)

    omega_y_b = M_b[1] / (1/12 * M * (H**2)) * dt
    omega_z_b = M_b[2] / (1/12 * M * (H**2)) * dt
    omega_x_b = 0

    vphi, vpsi, vgamma = omage_b_to_vattitude(omega_x_b, omega_y_b, omega_z_b, phi, psi, gamma)
    phi_new = phi + vphi * dt
    psi_new = psi + vpsi * dt
    gamma_new = gamma + vgamma * dt

    return phi_new, psi_new, gamma_new, vphi, vpsi, vgamma


def dynamic_thrust(state, action):
    theta_phi, theta_psi = state['theta_phi'], state['theta_psi']
    # F, vtheta_phi, vtheta_psi = action # for 1d exp

    F = state['F']
    vtheta_phi, vtheta_psi = 0, 0

    new_theta_phi = theta_phi + vtheta_phi * dt
    new_theta_psi = theta_psi + vtheta_psi * dt

    return F, new_theta_phi, new_theta_psi



def dynamic_step(state, action):

    already_landing = state['already_landing']
    if already_landing:
        vx, vy, vz, ax, ay, ax = 0, 0, 0, 0, 0, 0
        F, theta_phi, theta_psi = 0, 0, 0
    
    step_id = state['step_id'] + 1
    x_new, y_new, z_new, vx_new, vy_new, vz_new \
     = dynamic_centriod(state, action)
    phi_new, psi_new, gamma_new, vphi_new, vpsi_new, vgamma_new \
     = dynamic_attitude(state, action)
    F, theta_phi, theta_psi = dynamic_thrust(state, action)

    already_landing = check_landing_success(state)
    already_crash = check_crash(state)

   

    if already_crash or already_landing:
        done = True
    else:
        done = False

    x_new = round(x_new, 5) # 1d expr
    z_new = round(x_new, 5) # 1d expr
    state = {
        'x': x_new, 'y': y_new, 'z': z_new,
        'vx': vx_new, 'vy': vy_new, 'vz': vz_new,
        'phi': np.pi/2, 'psi': 0, 'gamma': 0, # for 1d expr
        'vphi': vphi_new, 'vpsi': vpsi_new, 'vgamma': vgamma_new,
        'F': F, 'theta_phi': theta_phi, 'theta_psi': theta_psi,
        'step_id': step_id, 't': step_id * dt,
        'already_crash': already_crash,
        'already_landing': already_landing
    }
    reward = calculate_reward(state)

    return state, reward, done, None
