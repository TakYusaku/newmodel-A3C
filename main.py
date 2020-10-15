import os
import numpy as np
import argparse
import gym
from gym import wrappers
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
import torch.multiprocessing as mp

from agent import Agent
from optim import AsyncRMSprop
from parameterserver import Policy
from environment import Environment
from wrapper_env import StackEnv, AtariEnv
from worker_thread import Worker_thread
import logger


if __name__ == '__main__':
    # hyperparameter の取得
    parser = argparse.ArgumentParser(description='PyTorch a3c')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor (default: 0.99)')
    parser.add_argument('--tau', type=float, default=1.00, metavar='T',
                        help='parameter for GAE (default: 1.00)')
    parser.add_argument('--seed', type=int, default=100, metavar='N',
                        help='random seed (default: 543)')
    parser.add_argument('--render', action='store_true',
                        help='render the environment')
    parser.add_argument('--monitor', action='store_true',
                        help='save the rendered video')
    parser.add_argument('--log_dir', type=str, default='./log_dir',
                        help='save dir')
    parser.add_argument('--epoch', type=int, default=100000, metavar='N',
                        help='training epoch number') #default=10000000
    parser.add_argument('--local_t_max', type=int, default=20, metavar='N',
                        help='bias variance control parameter')
    parser.add_argument('--entropy_beta', type=float, default=0.01, metavar='E',
                        help='coefficient of entropy')
    parser.add_argument('--v_loss_coeff', type=float, default=0.5, metavar='V',
                        help='coefficient of value loss')
    parser.add_argument('--frame_num', type=int, default=8, metavar='N',
                        help='number of frames you use as observation')
    parser.add_argument('--out_dim', type=int, default=128, metavar='N',
                        help='number of intermediate layer')
    parser.add_argument('--lr', type=float, default=7e-4, metavar='L',
                        help='learning rate')
    parser.add_argument('--env', type=str, default='CartPole-v0',
                        help='Environment')
    parser.add_argument('--atari', action='store_true',
                        help='atari environment')
    parser.add_argument('--num_process', type=int, default=1, metavar='n',
                        help='number of processes')
    parser.add_argument('--eps', type=float, default=0.01, metavar='E',
                        help='epsilon minimum log or std')
    parser.add_argument('--save_name', type=str, default='exp', metavar='N',
                        help='define save name')
    parser.add_argument('--save_mode', type=str, default='max', metavar='S',
                        help='save mode. all or last or max')
    args = parser.parse_args()

    
    logger.add_tabular_output(os.path.join(args.log_dir, 'progress.csv'))
    assert not (args.env == 'CartPole-v0' and not args.atari), 'You should use --atari option'
    logger.log_parameters_lite(os.path.join(args.log_dir, 'params.json'), args)
    

    env = gym.make(args.env)
    # 動画を保存するか
    if args.monitor:
        env = wrappers.Monitor(env, args.log_dir, force=True)
    env = StackEnv(env,args.frame_num)
    env.seed(args.seed)
    torch.manual_seed(args.seed)

    wt = Environment(env, args)

    # ディレクトリの作成
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_brain = Policy(env.action_space.n, dim_obs=env.observation_space.shape[0], out_dim=args.out_dim, frame_num=args.frame_num)#.to(device) # global brain の定義
    global_brain.share_memory()
    agent = Agent(env.action_space.n, dim_obs=env.observation_space.shape[0], out_dim=args.out_dim, frame_num=args.frame_num)    # 環境内で行動するagentを生成 (local brainに相当)
    #agent.local_brain.model.to(device)
    #agent.local_brain.model.share_memory()
    optimizer = AsyncRMSprop(global_brain.parameters(), agent.local_brain.parameters(), lr=args.lr, eps=args.eps)

    global_t = torch.LongTensor(1).share_memory_()
    global_t.zero_()
    processes = []

    pipe_reward = []
    for rank in range(args.num_process):
        get_rev,send_rev  = mp.Pipe(False)
        #p = mp.Process(target=wt.cartpole_train, args=(rank, env, global_brain, agent, optimizer, global_t, send_rev, args))
        p = mp.Process(target=wt.cartpole_train, args=(rank, env, global_brain, agent, optimizer, global_t, send_rev, args))
        processes.append(p)
        pipe_reward.append(get_rev)
        p.start()
    for p in processes:
        p.join()

    result_reward = pipe_reward[0].recv()
    for x in pipe_reward[1:]:
        result_reward += x.recv()

    result_reward = np.array(result_reward)
    sort_index = np.argsort(result_reward[:,0])

    sorted_result = result_reward[sort_index]

    
    plt.plot(range(len(sorted_result)), sorted_result[:,1])
    #plt.plot([0, len(global_history)], [195, 195], "--", color="darkred")
    plt.xlabel("episodes")
    plt.ylabel("Total Reward")
    plt.savefig("./log_dir/a3c_cartpole-v1.png")

    