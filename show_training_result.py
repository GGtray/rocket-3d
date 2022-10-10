import torch
from rocket import Rocket
from policy import ActorCritic
import os
import glob

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    task = 'landing'

    max_steps = 800
    ckpt_dir = glob.glob(os.path.join(task+'_ckpt', '*.pt'))
    # ckpt_dir = 'landing_ckpt/ckpt_00085001.pt'
    print(ckpt_dir)

    env = Rocket(task=task, max_steps=max_steps)
    net = ActorCritic(input_dim=env.state_dims, output_dim=env.action_dims).to(device)
    # print(type(env))
    # print(type(net))
    # print(env.state_dims, env.action_dims)

    checkpoint = torch.load(ckpt_dir)
    # print(checkpoint['model_G_state_dict'])
    net.load_state_dict(checkpoint['model_G_state_dict'])

    state = env.reset()
    # print(state)
    for step_id in range(max_steps):
        action, log_prob, value = net.get_action(state)
        state, reward, done, _ = env.step(action)
        env.render(window_name='fly')
        if env.already_crash:
            break