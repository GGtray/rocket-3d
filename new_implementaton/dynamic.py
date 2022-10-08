import random
import numpy as np

state_buffer = {}

dt = 0.05
g = 9.8
H = 50
I = 1/12 * H * H
target_r = 50

world_x_min = -300  # meters
world_x_max = 300
world_y_min = -30
world_y_max = 570


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


def calculate_reward(self, state):
        x_range = world_x_max - world_x_min
        y_range = world_y_max - world_y_min

        # dist between agent and target point
        dist_x = abs(state['x'] - self.target_x)
        dist_y = abs(state['y'] - self.target_y)
        dist_norm = dist_x / x_range + dist_y / y_range

        dist_reward = 0.1*(1.0 - dist_norm)

        if abs(state['theta']) <= np.pi / 6.0:
            pose_reward = 0.1
        else:
            pose_reward = abs(state['theta']) / (0.5*np.pi)
            pose_reward = 0.1 * (1.0 - pose_reward)

        reward = dist_reward + pose_reward

        v = (state['vx'] ** 2 + state['vy'] ** 2) ** 0.5
        
        if self.already_crash:
            reward = (reward + 5*np.exp(-1*v/10.)) * (self.max_steps - self.step_id)
        if self.already_landing:
            reward = (1.0 + 5*np.exp(-1*v/10.))*(self.max_steps - self.step_id)

        return reward


def init(state):
    # create random state
    x_range = world_x_max - world_x_min
    y_range = world_y_max - world_y_min
    xc = (world_x_max + world_x_min) / 2.0
    yc = (world_y_max + world_y_min) / 2.0

    x = xc
    y = yc + 0.2 * y_range
    theta = random.uniform(-45, 45) / 180 * np.pi
    vy = -10

    state = {
        'x': x, 'y': y, 
        'vx': 0, 'vy': vy,
        'theta': theta, 'vtheta': vtheta_new,
        'phi':0, 'f': 0,
        't': 0,
        'already_landing': False,
        'already_crash': False
    }
    
    return state

def step(state, action):
    

    # input last state
    x, y, z, vx, vy, vz = state['x'], state['y'], state['z'], state['vx'], state['vy'], state['vz'] # 质心运动

    # 绕质心运动
    theta, beta, gamma, vtheta, vbeta, vgamma \
    = state['theta'], state['beta'], state['gamma'], state['vtheta'], state['vbeta'], state['vgamma']

    # 是否落地了
    already_landing = state['already_landing']


    # input action
    f, vphi, psi = action['f'], action['vphi'], action['vpsi']

    ft, fr = -f*np.sin(phi), f*np.cos(phi)
    fx = ft*np.cos(theta) - fr*np.sin(theta)
    fy = ft*np.sin(theta) + fr*np.cos(theta)

    rho = 1 / (125/(g/2.0))**0.5  # suppose after 125 m free fall, then air resistance = mg
    ax, ay = fx-rho*vx, fy-g-rho*vy
    atheta = ft*H/2 / I


    #### update agent
    # check if landed first
    if already_landing:
        vx, vy, ax, ay, theta, vtheta, atheta = 0, 0, 0, 0, 0, 0, 0
        phi, f = 0, 0
        action = 0

    step_id += 1
    x_new = x + vx*dt + 0.5 * ax * (dt**2)
    y_new = y + vy*dt + 0.5 * ay * (dt**2)
    vx_new, vy_new = vx + ax * dt, vy + ay * dt
    theta_new = theta + vtheta*dt + 0.5 * atheta * (dt**2)
    vtheta_new = vtheta + atheta * dt
    phi = phi + dt*vphi

    phi = max(phi, -20/180*3.1415926)
    phi = min(phi, 20/180*3.1415926)                

    state = {
        'x': x_new, 'y': y_new, 'vx': vx_new, 'vy': vy_new,
        'theta': theta_new, 'vtheta': vtheta_new,
        'phi': phi, 'f': f,
        't': step_id * dt,
        'already_landing': already_landing
    }

    state_buffer.append(state)

    already_landing = check_landing_success(state)
    already_crash = check_crash(state)

    reward = calculate_reward(state)

    if self.already_crash or self.already_landing:
        done = True
    else:
        done = False

    return state, reward, done

