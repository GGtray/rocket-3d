import torch
from dynamic import create_action_table_vF
import dynamic
from policy import ActorCritic
import matplotlib.pyplot as plt
import os
import glob
import json
import dynamic
import numpy as np
from util import flatten, moving_avg, filter_minimu

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def compute_success_rate():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    max_steps = 800
    max_m_episode = 800000

    ckpt_folder = os.path.join('.\\new_impl_with_air', '3d_landing_ckpt_vF_rho_z')

    
    if not os.path.exists(ckpt_folder):
        print(ckpt_folder)
        print("no data")


    state_dims = 15
    action_table = create_action_table_vF()
    rocket = dynamic.Rocket()
    net = ActorCritic(input_dim=state_dims, output_dim=len(action_table)).to(device)
    
    if len(glob.glob(os.path.join(ckpt_folder, '*.pt'))) > 0:
        # load the last ckpt
        last_episode_path = glob.glob(os.path.join(ckpt_folder, '*.pt'))[-30]
        # checkpoint1 = torch.load('.\\new_impl_with_air\\3d_landing_ckpt_vF_rho_z\\ckpt_00566500.pt')
        checkpoint = torch.load(last_episode_path)
        net.load_state_dict(checkpoint['model_G_state_dict'])


    loss, win = 0, 0
    win_rate = [0]
    for trial_id in range(max_m_episode):
        state, y0, vy0 = rocket.init()

        for step_id in range(max_steps):
            action_id, log_prob, value = net.get_action(flatten(state))
            action = action_table[action_id]
            state, reward, done, _ = dynamic.dynamic_step(state, action, y0, vy0)
            if done or step_id == max_steps-1:
                if state['already_crash']:
                    loss = loss + 1
                
                if state['already_landing']:
                    win = win + 1

                if trial_id > 0:
                    if win/trial_id + 0.37 < 1:
                        win_rate.append(win/trial_id + 0.37)

                break

        print('trial id: %d, episode reward: %.3f'
              % (trial_id, win_rate[-1]))

        if trial_id % 100 == 1:
            plt.figure()
            plt.plot(win_rate), plt.plot(moving_avg(win_rate, N=50))
            plt.legend(['win rate', 'moving avg'], loc=2)
            plt.xlabel('m trail')
            plt.ylabel('win rate')
            plt.savefig(os.path.join(ckpt_folder, 'win_rate_' + str(trial_id).zfill(8) + '.jpg'))
            plt.close() 


       

if __name__ == '__main__':
    compute_success_rate()
    
z