import numpy as np

state_buffer = {}

def step(state, action):

        # input last state
        x, y, z, vx, vy, vz = state['x'], state['y'], state['z'], state['vx'], state['vy'], state['vz'] # 质心运动
        # 绕质心运动
        alpha, beta, gamma, valpha, vbeta, vgamma = state['alpha'], state['beta'], state['gamma'], state['valpha'], state['vbeta'], state['vgamma']


        # input action
        F, phi, psi = action['f'], action['phi'], action['psi']

        ax = -1 * F * (1/M) * (np.cos(alpha) * np.sin(phi) * np.cos(psi) + )

        # update agent
        if self.already_landing:
            vx, vy, ax, ay, theta, vtheta, atheta = 0, 0, 0, 0, 0, 0, 0
            phi, f = 0, 0
            action = 0

        self.step_id += 1

        x_new = x + vx*self.dt + 0.5 * ax * (self.dt**2)
        y_new = y + vy*self.dt + 0.5 * ay * (self.dt**2)
        vx_new, vy_new = vx + ax * self.dt, vy + ay * self.dt
        theta_new = theta + vtheta*self.dt + 0.5 * atheta * (self.dt**2)
        vtheta_new = vtheta + atheta * self.dt
        phi = phi + self.dt*vphi

        phi = max(phi, -20/180*3.1415926)
        phi = min(phi, 20/180*3.1415926)

        state = {
            'x': x_new, 'y': y_new, 'vx': vx_new, 'vy': vy_new,
            'theta': theta_new, 'vtheta': vtheta_new,
            'phi': phi, 'f': f,
            't': self.step_id, 'action_': action
        }
        state_buffer.append(self.state)

        self.already_landing = self.check_landing_success(self.state)
        self.already_crash = self.check_crash(self.state)
        reward = self.calculate_reward(self.state)

        if self.already_crash or self.already_landing:
            done = True
        else:
            done = False

        return state, reward, done

