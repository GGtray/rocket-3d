import torch
import dynamic
from policy import ActorCritic
import matplotlib.pyplot as plt
import os
import glob
import json
import dynamic
import numpy as np
from util import flatten, moving_avg


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



if __name__ == '__main__':

    max_steps = 800
    max_m_episode = 800000

    ckpt_folder = os.path.join('.\\new_implementation', 'landing_ckpt')

    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)

    
    state_dims = 8
    action_dims = 9

    rocket = dynamic.Rocket()
    net = ActorCritic(input_dim=state_dims, output_dim=action_dims).to(device)
 

    last_episode_id = 0
    REWARDS = []
    
    if len(glob.glob(os.path.join(ckpt_folder, '*.pt'))) > 0:
        # load the last ckpt
        checkpoint = torch.load(glob.glob(os.path.join(ckpt_folder, '*.pt'))[-1])

        net.load_state_dict(checkpoint['model_G_state_dict'])
        last_episode_id = checkpoint['episode_id']
        REWARDS = checkpoint['REWARDS']

    for episode_id in range(last_episode_id, max_m_episode):

        state = rocket.reset()
        rewards, log_probs, values, masks = [], [], [], []
        for step_id in range(max_steps):
            action, log_prob, value = net.get_action(flatten(state))
            state, reward, done, _ = dynamic.step(state, action)
            
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            masks.append(1-done)

            if done or step_id == max_steps-1:
                _, _, Qval = net.get_action(flatten(state))
                net.update_ac(net, rewards, log_probs, values, masks, Qval, gamma=0.999)
                break

        REWARDS.append(np.sum(rewards))
        print('episode id: %d, episode reward: %.3f'
              % (episode_id, np.sum(rewards)))

        if episode_id % 100 == 1:
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