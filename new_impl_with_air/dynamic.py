# from math import gamma
import random
import numpy as np

from dynamic_helper import M1_gamma, M2_psi, M3_phi, thrust_convert_phi_psi_to_1_2, thrust_to_fx_fy_fz, omage_b_to_vattitude, wind_v


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
world_y_max = 570 # 这个地方思考一下
world_z_min = -300
world_z_max = 300

max_steps = 800
target_x, target_z, target_y, target_r = 0, 0, H/2.0, 50

rho = M * 9.8 / (50)**2


def create_action_table_vF(
    vf0=0, vf1=0.1*9.8*M, vf2=-0.1*9.8*M, 
    vtheta_phi_0=0, vtheta_phi_1=5 / 180 * np.pi, vtheta_phi_2=-5 / 180 * np.pi,
    vtheta_psi_0=0, vtheta_psi_1=5 / 180 * np.pi, vtheta_psi_2=-5 / 180 * np.pi
    ):

    action_table = [
    [vf0, vtheta_phi_0, vtheta_psi_0], [vf0, vtheta_phi_1, vtheta_psi_0], [vf0, vtheta_phi_2, vtheta_psi_0],
    [vf1, vtheta_phi_0, vtheta_psi_0], [vf1, vtheta_phi_1, vtheta_psi_0], [vf1, vtheta_phi_2, vtheta_psi_0],
    [vf2, vtheta_phi_0, vtheta_psi_0], [vf2, vtheta_phi_1, vtheta_psi_0], [vf2, vtheta_phi_2, vtheta_psi_0],

    [vf0, vtheta_phi_0, vtheta_psi_1], [vf0, vtheta_phi_1, vtheta_psi_1], [vf0, vtheta_phi_2, vtheta_psi_1],
    [vf1, vtheta_phi_0, vtheta_psi_1], [vf1, vtheta_phi_1, vtheta_psi_1], [vf1, vtheta_phi_2, vtheta_psi_1],
    [vf2, vtheta_phi_0, vtheta_psi_1], [vf2, vtheta_phi_1, vtheta_psi_1], [vf2, vtheta_phi_2, vtheta_psi_1],
    
    [vf0, vtheta_phi_0, vtheta_psi_2], [vf0, vtheta_phi_1, vtheta_psi_2], [vf0, vtheta_phi_2, vtheta_psi_2],
    [vf1, vtheta_phi_0, vtheta_psi_2], [vf1, vtheta_phi_1, vtheta_psi_2], [vf1, vtheta_phi_2, vtheta_psi_2],
    [vf2, vtheta_phi_0, vtheta_psi_2], [vf2, vtheta_phi_1, vtheta_psi_2], [vf2, vtheta_phi_2, vtheta_psi_2],
    ]

    return action_table


def  create_random_start_state_vF():

        # predefined locations
        x_range = world_x_max - world_x_min
        y_range = world_y_max - world_y_min

        xc = (world_x_max + world_x_min) / 2.0
        yc = (world_y_max + world_y_min) / 2.0


        # x = random.uniform(xc - x_range / 6.0, xc + x_range / 6.0)
        x = 0
        z = 0
        y = yc + 0.4*y_range


        phi = random.uniform(np.pi/2 - 10/180 * np.pi, np.pi/2 + 10/180 * np.pi)
        psi = 0

        vy = -50

        F = 1.2 * 9.8 * M

        wind_vx, wind_vz = 0, 0

        state = {
            'x': x, 'y': y, 'z': z,
            'vx': 0, 'vy': vy, 'vz': 0,
            'phi': phi, 'psi': psi, 'gamma': 0,
            'vphi': 0, 'vpsi':0, 'vgamma': 0,
            'theta_phi': 0, 'theta_psi': 0, 'F': F,
            'wind_vx':wind_vx, 'wind_vz':wind_vz,
            't': 0, 'step_id': 0
        }

        return state, y, vy


def  create_random_start_state_vF_dir():

        # predefined locations
        x_range = world_x_max - world_x_min
        y_range = world_y_max - world_y_min
    

        xc = (world_x_max + world_x_min) / 2.0
        yc = (world_y_max + world_y_min) / 2.0


        x = random.uniform(xc, xc + x_range / 6.0)
        z = random.uniform(xc, xc + x_range / 6.0)
        y = yc + 0.4*y_range


        phi = random.uniform(np.pi/2, np.pi/2 + 10/180 * np.pi)
        if x < 0:
            phi = random.uniform(np.pi/2 - 10/180 * np.pi, np.pi/2)

        psi = random.uniform(0, np.pi/6)
        if z < 0:
            psi = -psi

        vy = -50

        F = 0.9 * 9.8 * M

        # wind_vx, wind_vz = 0, wind_v(y)
        wind_vx, wind_vz = 0, 0

        state = {
            'x': x, 'y': y, 'z': z,
            'vx': 0, 'vy': vy, 'vz': 0,
            'phi': phi, 'psi': psi, 'gamma': 0,
            'vphi': 0, 'vpsi':0, 'vgamma': 0,
            'theta_phi': 0, 'theta_psi': 0, 'F': F,
            'wind_vx':wind_vx, 'wind_vz':wind_vz,
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
        self.state, y0, vy0 = create_random_start_state_vF()
        self.state['already_landing'] = False
        self.state['already_crash'] = False
        return self.state, y0, vy0




def check_landing_success(state):
    x, y, z = state['x'], state['y'], state['z']
    vx, vy, vz = state['vx'], state['vy'], state['vz']
    phi, psi, gamma = state['phi'], state['psi'], state['gamma']
    vphi, vpsi, vgamma = state['vphi'], state['vpsi'], state['vgamma']


    v = (vx**2 + vy**2 + vz**2)**0.5
    return True \
        if y <= 0 + H / 2.0 \
        and v < 20.0 and np.sqrt(x**2 + z**2) < target_r \
        and abs(phi-np.pi/2) < 10/180*np.pi \
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
    if y <= 0 + H / 2.0 and v >= 20.0:
        crash = True
    if y <= 0 + H / 2.0 and np.sqrt(x**2 + z**2) >= target_r:
        crash = True
    if y <= 0 + H / 2.0 and  ((abs(phi- np.pi/2) >= 10/180*np.pi) or (abs(psi) >= 10/180*np.pi) or (abs(gamma) >= 10/180*np.pi)) :
        crash = True
    if y <= 0 + H / 2.0 and ((abs(vphi) >= 10/180*np.pi) or (abs(vpsi) >= 10/180*np.pi) or (abs(vgamma) >= 10/180*np.pi)) :
        crash = True
    return crash


def calculate_reward(state, y0, vy0):

        
        x_range = world_x_max - world_x_min
        y_range = world_y_max - world_y_min
        z_range = world_z_max - world_z_min

        # dist between agent and target point
        dist_x = abs(state['x'] - target_x)
        dist_y = abs(state['y'] - target_y)
        dist_z = abs(state['z'] - target_z)

        loc_norm = dist_x / x_range + dist_z / z_range + dist_y / y_range

        loc_reward = 0.1*(1.0 - loc_norm)

        dist_phi = abs(state['phi'] - np.pi/2)
        dist_psi = abs(state['psi'])
        dist_gamma = abs(state['gamma'])

        att_norm = dist_phi + dist_psi + dist_gamma

        if att_norm <= np.pi / 6.0:
            att_reward = 0.1
        else:
            att_reward = 0.1 * (1.0 - att_norm)

        vy_reward = 0
        if state['vy'] > 0:
            vy_reward = -0.1
        else:
            vy_norm = abs(state['vy']) / expect_vy(state['vy'], y0, vy0) 
            vy_reward = 0.1 * (1.0 - vy_norm)
            # 越接近0， 则vy应该越大（负数）， 所以越接近零的vy应该获得越多的奖励，当y = 0时，vy应该等于0，此时奖励最大
            # 也就是说，vy和y应该是正比，

        reward = loc_reward + att_reward + vy_reward

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
    F = state['F']
    wind_vx, wind_vz = state['wind_vx'], state['wind_vz']

    _, theta_phi, theta_psi = action


    theta_1, theta_2 = thrust_convert_phi_psi_to_1_2(theta_phi, theta_psi)
    fx, fy, fz = thrust_to_fx_fy_fz(F, theta_1, theta_2)
    f_b = [fx, fy, fz]

    # 当前姿态
    phi, psi, gamma = state['phi'], state['psi'], state['gamma']
    f_f = M3_phi(-phi, M2_psi(-psi, M1_gamma(-gamma, f_b)))
    G = [0, -g * M, 0]

    rho_vx = np.random.normal(wind_vx - vx, abs(wind_vx - vx)/10)
    rho_vz = np.random.normal(wind_vz - vz, abs(wind_vz - vz)/10)


    # f_rho = [rho*rho_vx, 0, rho*rho_vz]

    f_rho = 0


    f_f_joint = f_f + G + f_rho
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
    F = state['F']
    vF, theta_phi, theta_psi = action # for 1d expr


    theta_1, theta_2 = thrust_convert_phi_psi_to_1_2(theta_phi, theta_psi)
    fx, fy, fz = thrust_to_fx_fy_fz(F, theta_1, theta_2)
    f_b = [fx, fy, fz]
    r = [-H/2, 0, 0]
    M_b = np.cross(r, f_b)

    omega_y_b = M_b[1] / (1/12 * M * (H**2)) * dt
    omega_z_b = M_b[2] / (1/12 * M * (H**2)) * dt
    omega_x_b = M_b[0] / (1/12 * M * (H**2)) * dt # 它如果不是零，动力学肯定是错的

    vphi, vpsi, vgamma = omage_b_to_vattitude(omega_x_b, omega_y_b, omega_z_b, phi, psi, gamma)
    phi_new = phi + vphi * dt
    psi_new = psi + vpsi * dt
    gamma_new = gamma + vgamma * dt

    return phi_new, psi_new, gamma_new, vphi, vpsi, vgamma


def dynamic_thrust_vF(state, action):
    F, theta_phi, theta_psi = state['F'], state['theta_phi'], state['theta_psi']
    
    vF, vtheta_phi, vtheta_psi = action # for 1d exp
    # F = state['F']
    # vtheta_phi, vtheta_psi = 0, 0

    new_theta_phi = theta_phi + vtheta_phi * dt
    new_theta_psi = theta_psi + vtheta_psi * dt
    new_F = F + vF

    if new_theta_phi < -25/180 * np.pi: new_theta_phi = -25/180 * np.pi
    elif new_theta_phi > 25/180 * np.pi: new_theta_phi = 25/180 * np.pi
    
    if new_theta_psi < -25/180 * np.pi: new_theta_psi = -25/180 * np.pi
    elif new_theta_psi > 25/180 * np.pi: new_theta_psi = 25/180 * np.pi

    if new_F < 0.2 * 9.8 * M: new_F = 0.2 * 9.8 * M
    elif new_F > 2 * 9.8 * M: new_F = 2 * 9.8 * M

    return new_F, new_theta_phi, new_theta_psi




def dynamic_step(state, action, y0, vy0):

    already_landing = state['already_landing']
    if already_landing:
        vx, vy, vz, ax, ay, ax = 0, 0, 0, 0, 0, 0
        F, theta_phi, theta_psi = 0, 0, 0


    step_id = state['step_id'] + 1
    x_new, y_new, z_new, vx_new, vy_new, vz_new \
     = dynamic_centriod(state, action)
    phi_new, psi_new, gamma_new, vphi_new, vpsi_new, vgamma_new \
     = dynamic_attitude(state, action)
    F, theta_phi, theta_psi = dynamic_thrust_vF(state, action)

    already_landing = check_landing_success(state)
    already_crash = check_crash(state)

   

    if already_crash or already_landing:
        done = True
    else:
        done = False

    state = {
        'x': x_new, 'y': y_new, 'z': z_new,
        'vx': vx_new, 'vy': vy_new, 'vz': vz_new,
        'phi': phi_new, 'psi': psi_new, 'gamma': gamma_new, # for 1d expr
        'vphi': vphi_new, 'vpsi': vpsi_new, 'vgamma': vgamma_new,
        'F': F, 'theta_phi': theta_phi, 'theta_psi': theta_psi,
        'step_id': step_id, 't': step_id * dt,
        'wind_vx': 0, 'wind_vz': wind_v(y_new), 
        'already_crash': already_crash,
        'already_landing': already_landing,
    }
    reward = calculate_reward(state, y0, vy0)

    return state, reward, done, None


def expect_vy(y, y0, vy0):
    a = vy0 ** 2 / (2*y0) # a > 0
    # print('a', a)
    t_e = abs(vy0) / a # t_e > 0
    # print('t_e',t_e)

    t = t_e - (2 * abs(y) / a) ** 0.5# t > 0
    # print('t', t)
    abs_vy = abs(vy0) - a * t
    return abs_vy

if __name__ == "__main__":

    print('vy', expect_vy(0, 600, -50))
    pass 
