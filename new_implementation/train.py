import torch
from dynamic import create_action_table_vF, create_action_table
import dynamic
from dynamic import create_action_table_1d
from policy import ActorCritic
import matplotlib.pyplot as plt
import os
import glob
import json
import dynamic
import numpy as np
from util import flatten, moving_avg



def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    max_steps = 800
    max_m_episode = 800000

    ckpt_folder = os.path.join('.\\new_implementation', '3d_landing_ckpt_vF')
    result_array_folder = os.path.join('.\\new_implementation', '3d_trajectory_vF')
    if not os.path.exists(result_array_folder):
        os.mkdir(result_array_folder)
    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)

    
    state_dims = 15

    action_table = create_action_table_vF()
    rocket = dynamic.Rocket()
    net = ActorCritic(input_dim=state_dims, output_dim=len(action_table)).to(device)
 

    last_episode_id = 0
    REWARDS = []
    
    if len(glob.glob(os.path.join(ckpt_folder, '*.pt'))) > 0:
        # load the last ckpt
        checkpoint = torch.load(glob.glob(os.path.join(ckpt_folder, '*.pt'))[-1])

        net.load_state_dict(checkpoint['model_G_state_dict'])
        last_episode_id = checkpoint['episode_id']
        REWARDS = checkpoint['REWARDS']

    
    for episode_id in range(last_episode_id, max_m_episode):
        state_buffer = []
        state = rocket.init()
        rewards, log_probs, values, masks = [], [], [], []
        for step_id in range(max_steps):
            action_id, log_prob, value = net.get_action(flatten(state))
            action = action_table[action_id]
            state, reward, done, _ = dynamic.dynamic_step(state, action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            masks.append(1-done)
            state_buffer.append(state)
            if done or step_id == max_steps-1:
                _, _, Qval = net.get_action(flatten(state))
                net.update_ac(net, rewards, log_probs, values, masks, Qval, gamma=0.999)
                break

        REWARDS.append(np.sum(rewards))
        print('episode id: %d, episode reward: %.3f'
              % (episode_id, np.sum(rewards)))

        if episode_id % 100 == 0:
            plt.figure()
            plt.plot(REWARDS), plt.plot(moving_avg(REWARDS, N=50))
            plt.legend(['episode reward', 'moving avg'], loc=2)
            plt.xlabel('m episode')
            plt.ylabel('reward')
            plt.savefig(os.path.join(ckpt_folder, 'rewards_' + str(episode_id).zfill(8) + '.jpg'))
            plt.close() 

            torch.save({'episode_id': episode_id,
                        'REWARDS': REWARDS,
                        'model_G_state_dict': net.state_dict()},
                        os.path.join(ckpt_folder, 'ckpt_' + str(episode_id).zfill(8) + '.pt'))
            
            json_array = json.dumps(state_buffer)
            with open(result_array_folder + '\\{episode_id}.json'.format(episode_id=episode_id), 'w') as f:
                f.write(json_array)
            
            # result_view_table(json_array)

if __name__ == '__main__':
    train()
