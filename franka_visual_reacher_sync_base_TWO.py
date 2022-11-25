import torch
import argparse
import relod.utils as utils
import time
import numpy as np
import cv2
import os

from relod.logger import Logger
from relod.algo.comm import MODE
from relod.algo.local_wrapper import LocalWrapper
from relod.algo.sac_rad_agent import SACRADLearner, SACRADPerformer
from relod.envs.visual_franka_rl655_reacher.env import FrankaPanda_Visual_Reacher_RL655

config = {
    
    'conv': [
        # in_channel, out_channel, kernel_size, stride
        [-1, 32, 3, 2],
        [32, 32, 3, 2],
        [32, 32, 3, 2],
        [32, 32, 3, 1],
    ],
    
    'latent': 50,

    'mlp': [
        [-1, 1024], # first hidden layer
        [1024, 1024], 
        [1024, -1] # output layer
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(description='Local Franka Visual Synchronous Baseline')
    # environment
    parser.add_argument('--setup', default='Visual-Franka')
    parser.add_argument('--env', default='franka_visual_reacher_sync_base', type=str) 
    parser.add_argument('--camera_id', default=0, type=int)
    parser.add_argument('--image_width', default=160, type=int)
    parser.add_argument('--image_height', default=90, type=int)
    parser.add_argument('--target_type', default='size', type=str)
    parser.add_argument('--random_action_repeat', default=1, type=int)
    parser.add_argument('--agent_action_repeat', default=1, type=int)
    parser.add_argument('--image_history', default=3, type=int)
    parser.add_argument('--joint_history', default=1, type=int)
    parser.add_argument('--ignore_joint', default=False, action='store_true')
    parser.add_argument('--episode_length_time', default=6.0, type=float)
    parser.add_argument('--dt', default=0.075, type=float)

    parser.add_argument('--size_tol', default=0.5, type=float)
    parser.add_argument('--center_tol', default=0.1, type=float)
    parser.add_argument('--reward_tol', default=2.0, type=float)
    parser.add_argument('--reset_penalty_steps', default=70, type=int)
    parser.add_argument('--reward', default=-1, type=float)
    parser.add_argument('--reward_scale', default=(75.0/40.0), type=float)
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=100000, type=int)
    parser.add_argument('--rad_offset', default=0.01, type=float)
    # train
    parser.add_argument('--init_steps', default=2666, type=int) 
    parser.add_argument('--env_steps', default=57600, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--async_mode', default=False, action='store_true')
    parser.add_argument('--max_updates_per_step', default=1.0, type=float)
    parser.add_argument('--update_every', default=1, type=int)
    parser.add_argument('--update_epochs', default=1, type=int)
    # critic
    parser.add_argument('--critic_lr', default=3e-4, type=float)
    parser.add_argument('--critic_tau', default=0.005, type=float)
    parser.add_argument('--critic_target_update_freq', default=1, type=int)
    parser.add_argument('--bootstrap_terminal', default=0, type=int)
    # actor
    parser.add_argument('--actor_lr', default=3e-4, type=float)
    parser.add_argument('--actor_update_freq', default=1, type=int)
    # encoder
    parser.add_argument('--encoder_tau', default=0.005, type=float)
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=3e-4, type=float)
    # agent
    parser.add_argument('--remote_ip', default='localhost', type=str)
    parser.add_argument('--port', default=9876, type=int)
    parser.add_argument('--mode', default='l', type=str, help="Modes in ['r', 'l', 'rl', 'e'] ")
    # misc
    parser.add_argument('--description', default='test new remote script', type=str)
    parser.add_argument('--seed', default=4, type=int)
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_model', default=True, action='store_true')
    parser.add_argument('--plot_learning_curve', default=False, action='store_true')
    parser.add_argument('--xtick', default=10, type=int)
    parser.add_argument('--display_image', default=False, action='store_true')
    parser.add_argument('--save_image', default=False, action='store_true')
    parser.add_argument('--save_model_freq', default=10000, type=int)
    parser.add_argument('--load_model', default=-1, type=int)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--lock', default=False, action='store_true')

    args = parser.parse_args()
    assert args.mode in ['r', 'l', 'rl', 'e']
    assert args.reward < 0 and args.reset_penalty_steps >= 0

    return args

def main():
    args = parse_args()

    if args.mode == 'r':
        mode = MODE.REMOTE_ONLY
    elif args.mode == 'l':
        mode = MODE.LOCAL_ONLY
    elif args.mode == 'rl':
        mode = MODE.REMOTE_LOCAL
    elif args.mode == 'e':
        mode = MODE.EVALUATION
    else:
        raise  NotImplementedError()

    assert args.async_mode == False

    if args.device is '':
        args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    args.work_dir += f'/results/{args.env}/seed={args.seed}'
    args.model_dir = args.work_dir+'/models'
    args.return_dir = args.work_dir+'/returns'

    os.makedirs(args.model_dir, exist_ok=False)
    os.makedirs(args.return_dir, exist_ok=False)

    rewards_path = args.work_dir+'/rewards.txt'

    # sample_time_path = args.work_dir+'/sample_time.txt'
    # env_time_path = args.work_dir+'/env_time.txt'
    agent_env_time_path = args.work_dir+'/agent_env_time.txt'
    batch_time_path = args.work_dir+'/batch_time.txt'
    update_time_path = args.work_dir+'/update_time.txt'
    num_updates_path = args.work_dir+'/num_updates.txt'

    # sample_time_fl = open(sample_time_path, "w+")
    # env_time_fl = open(env_time_path, "w+")
    agent_env_time_fl = open(agent_env_time_path, "w+")
    rewards_fl = open(rewards_path, "w+")
    batch_time_fl = open(batch_time_path, "w+")
    update_time_fl= open(update_time_path, "w+")
    num_updates_fl = open(num_updates_path, "w+")

    if mode == MODE.LOCAL_ONLY:
        L = Logger(args.return_dir, use_tb=args.save_tb)

    # if args.save_image:
    #     args.image_dir = args.work_dir+'/images'
    #     if mode == MODE.LOCAL_ONLY:
    #         os.makedirs(args.image_dir, exist_ok=False)

    episode_length_step = int(args.episode_length_time / args.dt)

    env = FrankaPanda_Visual_Reacher_RL655(
        dt=args.dt,
        image_width=args.image_width,
        image_height=args.image_height,
        camera_index=args.camera_id, 
        episode_steps=episode_length_step,
        size_tol=args.size_tol)

    utils.set_seed_everywhere(args.seed, env)

    image, prop = env.reset()

    #print(args.reward_scale)
    
    input('go?')
    
    args.image_shape = env.image_space.shape
    args.proprioception_shape = env.observation_space.shape

    print(args.image_shape, image.shape)
    print(args.proprioception_shape, prop.shape)

    args.action_shape = env.action_space.shape
    args.env_action_space = env.action_space
    args.net_params = config

    
    agent = LocalWrapper(episode_length_step, mode, remote_ip=args.remote_ip, port=args.port)
    agent.send_data(args)
    agent.init_performer(SACRADPerformer, args)
    agent.init_learner(SACRADLearner, args, agent.performer)

    # sync initial weights with remote
    agent.apply_remote_policy(block=True)

    if args.load_model > -1:
        agent.load_policy_from_file(args.model_dir, args.load_model)
    
    # First inference took a while (~1 min), do it before the agent-env interaction loop
    if mode != MODE.REMOTE_ONLY:
        agent.performer.sample_action((image, prop))
        agent.performer.sample_action((image, prop))
        agent.performer.sample_action((image, prop))

    # Experiment block starts
    experiment_done = False
    total_steps = 0
    sub_epi = 0
    num_updates = 0
    returns = []
    epi_lens = []
    start_time = time.time()
    print(f'Experiment starts at: {start_time}')

    while not experiment_done:
        image, prop = env.reset() 
        agent.send_init_ob((image, prop))
        ret = 0
        epi_steps = 0
        epi_done = 0
        rewards = []
        # if mode == MODE.LOCAL_ONLY and args.save_image:
        #     episode_image_dir = args.image_dir+f'/episode={len(returns)+1}/'
        #     os.makedirs(episode_image_dir, exist_ok=False)

        epi_start_time = time.time()

        while not experiment_done and not epi_done:
            # if args.display_image or (mode == MODE.LOCAL_ONLY and args.save_image):
            #     image_to_show = np.transpose(image, [1, 2, 0])
            #     image_to_show = image_to_show[:,:,-3:]
            #     if mode == MODE.LOCAL_ONLY and args.save_image:
            #         cv2.imwrite(episode_image_dir+f'{epi_steps}.png', image_to_show)
            #     if args.display_image:
            #         cv2.imshow('raw', image_to_show)
            #         cv2.waitKey(1)

            # select an action
            agent_env_time = 0
            t1 = time.time()
            action = agent.sample_action((image, prop))
            agent_env_time = time.time() - t1

            
            # step in the environment
            next_image, next_prop, reward, epi_done, _ = env.step(action)
            reward = reward * args.reward_scale
            # store
            
            t1 = time.time()
            agent.push_sample((image, prop), action, reward, (next_image, next_prop), epi_done)
            agent_env_time += (time.time() - t1)

            stat, batch_tm, update_tm = agent.update_policy(total_steps)
            if stat is not None:
                num_updates = stat['train/num_updates']

           
            if batch_tm is not None:
                agent_env_time_fl.write(str(agent_env_time)+'\n')
                # env_time_fl.write(str(env_time)+'\n')
                # sample_time_fl.write(str(sample_time)+'\n')
                batch_time_fl.write(str(batch_tm)+'\n')
                update_time_fl.write(str(update_tm)+'\n')
                

            image = next_image
            prop = next_prop

            # Log
            total_steps += 1
            ret += reward
            epi_steps += 1
            rewards.append(reward)

            if args.save_model and total_steps % args.save_model_freq == 0:
                agent.save_policy_to_file(args.model_dir, total_steps)

            # if not epi_done and sub_steps >= episode_length_step: # set timeout here
            #     sub_steps = 0
            #     sub_epi += 1
            #     t1 = time.time()
            #     image, prop = env.reset()
            #     reset_step = (time.time() - t1) // args.dt
            #     ret += reset_step  * args.reward
            #     print(f'Sub episode {sub_epi} done.')
            #     agent.send_init_ob((image, prop))

            experiment_done = total_steps >= args.env_steps

        # save the last image
        # if mode == MODE.LOCAL_ONLY and args.save_image:
        #     image_to_show = np.transpose(image, [1, 2, 0])
        #     image_to_show = image_to_show[:,:,-3:]
        #     cv2.imwrite(episode_image_dir+f'{epi_steps}.png', image_to_show)

        fl = open("/home/franka/project/ReLoDFranka/rl655/end.txt", "r")
        en = int(fl.read())
        fl.close()
        if en == 1:
            break

        if epi_done: # episode done, save result
            returns.append(ret)
            epi_lens.append(epi_steps)
            rewards_fl.write(str(rewards)+'\n\n')
            utils.save_returns(args.return_dir+'/return.txt', returns, epi_lens)

            wr_st = "Episdoe: " + str(len(returns)) + ",  Total Steps: " + str(total_steps) + ",  Num Updates: " + str(num_updates) + "\n"
            num_updates_fl.write(wr_st)
            
            if mode == MODE.LOCAL_ONLY:
                L.log('train/duration', time.time() - epi_start_time, total_steps)
                L.log('train/episode_reward', ret, total_steps)
                L.log('train/episode', len(returns), total_steps)
                L.dump(total_steps)
                if args.plot_learning_curve:
                    utils.show_learning_curve(args.return_dir+'/learning curve.png', returns, epi_lens, xtick=args.xtick)

    duration = time.time() - start_time
    agent.save_policy_to_file(args.model_dir, total_steps)

    # Clean up

    # sample_time_fl.close()
    # env_time_fl.close()
    num_updates_fl.close()
    agent_env_time_fl.close()
    batch_time_fl.close()
    update_time_fl.close()
    rewards_fl.close()
    env.reset()
    agent.close()
    env.close()

    # always show a learning curve at the end
    if mode == MODE.LOCAL_ONLY:
        utils.show_learning_curve(args.return_dir+'/learning curve.png', returns, epi_lens, xtick=args.xtick)
    print(f"Finished in {duration}s")

if __name__ == '__main__':
    main()
