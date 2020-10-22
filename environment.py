## task
# global frames をどうするか
# global isLearned をどうするか
# l.34周辺 の 動画保存をどうするか
# TmaxとENVをどこに突っ込むか

import os
import numpy as np
import gym
import time
from collections import deque
import torch
import torch.autograd as autograd
from torch.autograd import Variable

from agent import Agent 
from optim import AsyncRMSprop
from parameterserver import Policy
from wrapper_env import StackEnv, AtariEnv
import logger

#ENV = 'CartPole-v0'
Tmax = 10
NUM_ACTIONS = 4 #env.action_space.n

'''
@dataclass
class Step:
    value: float
    prob: float
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
'''


# --CartPoleを実行する環境 -------
class Environment:
    total_reward_vec = np.zeros(10)  # 総報酬を10試行分格納して、平均総報酬をもとめる
    count_trial_each_thread = 0     # 各環境の試行数

    def __init__(self, args, global_ep, global_ep_r, res_queue, global_brain, optimizer):
        self.global_history_reward = []
        #self.total_reward = 0
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.global_brain, self.optimizer = global_brain, optimizer
        self.env = gym.make('CartPole-v0').unwrapped
        self.env = StackEnv(self.env,args.frame_num)
        self.env.seed(args.seed)
        self.agent = Agent(self.env.action_space.n, dim_obs=self.env.observation_space.shape[0], out_dim=args.out_dim, frame_num=args.frame_num)    # 環境内で行動するagentを生成 (local brainに相当)           # local network

    def cartpole_train_3_5(self, rank, args):
        torch.manual_seed(args.seed + rank)

        self.agent.local_brain.train()

        step = 0
        sum_rewards = 0
        max_sum_rewards = 0
        vs = []
        entropies = []
        cnt = 0

        while self.g_ep.value < args.epoch:
            #tmp = 0
            o = self.env.reset()
            #o = torch.from_numpy(state)
            #print('cnt:',cnt)
            # self.agent.local_brain.sync(self.global_brain) # local policy にコピー
            observations, actions, values, rewards, probs = [], [], [], [], []
            #R = 0
            #done = True
            ep_r = 0.
            while True:
                step += 1
                # Agentのactで行動を取得
                p, v = self.agent.local_brain(Variable(torch.from_numpy(o).float()).unsqueeze(0))
                a = self.agent.act(o)
                if len(a.data.squeeze().size()) == 0:
                    o, r, done, _ = self.env.step(a.data.squeeze().item())
                else:
                    o, r, done, _ = self.env.step(a.data.squeeze()[0])
                if done: r = -1
                if rank == 0:
                    sum_rewards += r
                    if args.render:
                        self.env.render()
                ep_r += r
                observations.append(o)
                actions.append(a)
                values.append(v)
                rewards.append(r)
                probs.append(p)

                if step % args.local_t_max == 0 or done:
                    if done:
                        R = 0
                    else:
                        _, v = self.agent.local_brain(torch.from_numpy(observations[-1]).unsqueeze(0).float())
                        R = v.data.squeeze().item()

                    returns = []
                    for r in rewards[::-1]: # 割引報酬和
                        R = r + 0.99 * R
                        returns.insert(0, R)
                    returns = torch.Tensor(returns)


                    loss, v_loss, entropy, _ = self.agent._loss_function(actions, values, probs, returns, args)
                    vs.append(v_loss.data.numpy())
                    entropies.append(entropy.data.numpy())

                    ## 記録
                    if rank == 0 and done:
                        logger.record_tabular_misc_stat('Entropy', entropies)
                        logger.record_tabular_misc_stat('V', vs)
                        logger.record_tabular('reward', sum_rewards)
                        logger.record_tabular('step', self.g_ep.value)
                        logger.dump_tabular()
                        del vs[:]
                        del entropies[:]
                    self.optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    for lp, gp in zip(self.agent.local_brain.parameters(), self.global_brain.parameters()):
                        gp._grad = lp.grad

                    self.optimizer.step()
                    self.agent.local_brain.sync(self.global_brain) # local policy にコピー

                    observations, actions, values, rewards, probs = [], [], [], [], []

                if done:
                    with self.g_ep.get_lock():
                        self.g_ep.value += 1
                    with self.g_ep_r.get_lock():
                        if self.g_ep_r.value == 0.:
                            self.g_ep_r.value = ep_r
                        else:
                            self.g_ep_r.value = self.g_ep_r.value * 0.99 + ep_r * 0.01
                    self.res_queue.put(self.g_ep_r.value)

                    o = self.env.reset()
                    #self.global_history_reward.append([tmp, self.total_reward])
                    self.total_reward = 0
                    if rank == 0:
                        print('----------------------------------')
                        print('total reward of the episode:', sum_rewards)
                        print('----------------------------------')
                        if args.save_mode == 'all':
                            torch.save(self.agent.local_brain, os.path.join(args.log_dir, args.save_name+"_{}.pkl".format(self.g_ep.value)))
                        elif args.save_mode == 'last':
                            torch.save(self.agent.local_brain, os.path.join(args.log_dir, args.save_name+'.pkl'))
                        elif args.save_mode == 'max':
                            if max_sum_rewards < sum_rewards:
                                torch.save(self.agent.local_brain, os.path.join(args.log_dir, args.save_name+'.pkl'))
                                max_sum_rewards = sum_rewards
                        #step = 0
                        sum_rewards = 0
                    break

            #raise
            # 学習率の更新
            # new_lr = np.true_divide(args.epoch - global_t[0] , args.epoch * args.lr)
            # self.optimizer.step(new_lr)

            cnt += 1

        #send_rev.send(self.global_history_reward)
        self.res_queue.put(None)

    '''
    def cartpole_train_3(self, rank, args, global_brain, send_rev, optimizer=None):
        torch.manual_seed(args.seed + rank)

        env = gym.make(args.env)
        env = StackEnv(env,args.frame_num)
        env.seed(args.seed + rank)

        agent = Agent(env.action_space.n, dim_obs=env.observation_space.shape[0], out_dim=args.out_dim, frame_num=args.frame_num)    # 環境内で行動するagentを生成 (local brainに相当)
        #model = ActorCritic(env.observation_space.shape[0], env.action_space)

        if optimizer is None:
            optimizer = optim.Adam(global_brain.parameters(), lr=args.lr)

        agent.local_brain.train()

        state = env.reset()
        state = torch.from_numpy(state)
        done = True

        episode_length = 0
        while True:
            episode_length += 1
            # Sync with the shared model every iteration
            agent.local_brain.load_state_dict(global_brain.state_dict())
    '''
    '''
            if done:
            # initialization
                cx = Variable(torch.zeros(1, 128))
                hx = Variable(torch.zeros(1, 128))
            else:
                cx = Variable(cx.data)
                hx = Variable(hx.data)
    '''
    '''
            values = []
            log_probs = []
            rewards = []
            entropies = []

            for step in range(args.num_steps):
                # for mujoco, env returns DoubleTensor
                p, v = agent.local_brain((Variable(state.float().unsqueeze(0).float())))
                sigma_sq = F.softplus(sigma_sq)
                eps = torch.randn(mu.size())
                # calculate the probability
                action = (mu + sigma_sq.sqrt()*Variable(eps)).data
                prob = normal(action, mu, sigma_sq)
                entropy = -0.5*((sigma_sq+2*pi.expand_as(sigma_sq)).log()+1)

                entropies.append(entropy)
                log_prob = prob.log()

                state, reward, done, _ = env.step(action.numpy())
            # prevent stuck agents
                done = done or episode_length >= args.max_episode_length
            # reward shaping
                reward = max(min(reward, 1), -1)

                if done:
                    episode_length = 0
                    state = env.reset()

                state = torch.from_numpy(state)
                values.append(value)
                log_probs.append(log_prob)
                rewards.append(reward)

                if done:
                    break

            R = torch.zeros(1, 1)
            if not done:
                value, _, _, _ = model((Variable(state.float().unsqueeze(0)), (hx, cx)))
                R = value.data

            values.append(Variable(R))
            policy_loss = 0
            value_loss = 0
            R = Variable(R)
            gae = torch.zeros(1, 1)
        # calculate the rewards from the terminal state
            for i in reversed(range(len(rewards))):
                R = args.gamma * R + rewards[i]
                advantage = R - values[i]
                value_loss = value_loss + 0.5 * advantage.pow(2)

                # Generalized Advantage Estimataion
            # convert the data into xxx.data will stop the gradient
                delta_t = rewards[i] + args.gamma * \
                    values[i + 1].data - values[i].data
                gae = gae * args.gamma * args.tau + delta_t

            # for Mujoco, entropy loss lower to 0.0001
                policy_loss = policy_loss - (log_probs[i]*Variable(gae).expand_as(log_probs[i])).sum() \
                        - (0.0001*entropies[i]).sum()

            optimizer.zero_grad()

            (policy_loss + 0.5 * value_loss).backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 40)

            ensure_shared_grads(model, global_brain)
            optimizer.step()
    '''

    def cartpole_train_2(self, rank, env, global_brain, agent, optimizer, global_t, send_rev, args):
        #global_total_loss = []
        o = env.reset()
        step = 0
        sum_rewards = 0
        max_sum_rewards = 0
        vs = []
        entropies = []
        sum_rewards = 0
        done = True
        #cnt = 0
        while global_t[0] < args.epoch:
            tmp = global_t.clone().item() + 1
            #print('cnt:',cnt)
            agent.local_brain.sync(global_brain) # local policy にコピー
            observations = []
            actions = []
            values = []
            rewards = []
            probs = []
            R = 0
            for _ in range(args.local_t_max):
                global_t += 1
                step += 1
                # Agentのactで行動を取得
                p, v = agent.local_brain(Variable(torch.from_numpy(o).float()).unsqueeze(0))
                a = agent.act(o)
                if len(a.data.squeeze().size()) == 0:
                    o, r, done, _ = env.step(a.data.squeeze().item())
                else:
                    o, r, done, _ = env.step(a.data.squeeze()[0])
                r = max(min(r, 1), -1)
                sum_rewards += r
                if rank == 0:
                    if args.render:
                        env.render()

                observations.append(o)
                actions.append(a)
                values.append(v)
                rewards.append(r)
                probs.append(p)
                if done:
                    o = env.reset()
                    #self.total_reward = 0
                    if rank == 0:
                        print('----------------------------------')
                        print('total reward of the episode:', sum_rewards)
                        print('----------------------------------')
                        if args.save_mode == 'all':
                            torch.save(agent.local_brain, os.path.join(args.log_dir, args.save_name+"_{}.pkl".format(global_t[0])))
                        elif args.save_mode == 'last':
                            torch.save(agent.local_brain, os.path.join(args.log_dir, args.save_name+'.pkl'))
                        elif args.save_mode == 'max':
                            if max_sum_rewards < sum_rewards:
                                torch.save(agent.local_brain, os.path.join(args.log_dir, args.save_name+'.pkl'))
                                max_sum_rewards = sum_rewards
                        step = 0
                    break
                '''
                else:
                    #self.total_reward += r
                    _, v = agent.local_brain(torch.from_numpy(o).unsqueeze(0).float())
                    R += v.data.squeeze().item()
                '''

            # -- Agent advantage_push_agent.local_brain() --- 割引報酬和の計算
            '''
            returns = []
            for r in rewards[::-1]: # 割引報酬和
                R = r + 0.99 * R
                returns.insert(0, R)
            returns = torch.Tensor(returns)


            #if len(returns) > 1:
            #    returns = (returns-returns.mean()) / (returns.std()+args.eps)

            # -- LocalBrain _build_graph() --- lossの計算

            loss, v_loss, entropy, p_loss_list = agent._loss_function(actions, values, probs, returns, args)
            '''
            if not done:
                _, v = agent.local_brain(torch.from_numpy(o).unsqueeze(0).float())
                R = v.data.squeeze().item()
            #print('R',R,type(R))

            values.append(torch.tensor([[R]]))
            loss, v_loss, entropy, p_loss_list = agent._loss_function_2(actions, values, probs, rewards, R, args)

            vs.append(v_loss.data.numpy())
            entropies.append(entropy.data.numpy())

            self.global_history_reward.append([tmp, sum_rewards])

            ## 記録
            if rank == 0 and done:
                logger.record_tabular_misc_stat('Entropy', entropies)
                logger.record_tabular_misc_stat('V', vs)
                logger.record_tabular('reward', sum_rewards)
                logger.record_tabular('step', global_t[0])
                logger.dump_tabular()
                del vs[:]
                del entropies[:]
            sum_rewards = 0

            # 重みの更新(最後まで)
            optimizer.zero_grad()
            final_node = [loss] + p_loss_list
            gradients = [torch.ones(1)] + [None] * len(p_loss_list)
            autograd.backward(final_node, gradients)
            # 学習率の更新
            new_lr = np.true_divide(args.epoch - global_t[0] , args.epoch * args.lr)
            optimizer.step(new_lr)

            # cnt += 1

        send_rev.send(self.global_history_reward)
#'''
    def cartpole_train(self, rank, env, global_brain, agent, optimizer, global_t, send_rev, args):
        #global_total_loss = []
        o = env.reset()
        step = 0
        sum_rewards = 0
        max_sum_rewards = 0
        vs = []
        entropies = []
        sum_rewards = 0
        done = True
        #cnt = 0
        while global_t[0] < args.epoch:
            tmp = global_t.clone().item() + 1
            #print('cnt:',cnt)
            agent.local_brain.sync(global_brain) # local policy にコピー
            observations = []
            actions = []
            values = []
            rewards = []
            probs = []
            R = 0
            for _ in range(args.local_t_max):
                global_t += 1
                step += 1
                # Agentのactで行動を取得
                p, v = agent.local_brain(Variable(torch.from_numpy(o).float()).unsqueeze(0))
                a = agent.act(o)
                if len(a.data.squeeze().size()) == 0:
                    o, r, done, _ = env.step(a.data.squeeze().item())
                else:
                    o, r, done, _ = env.step(a.data.squeeze()[0])
                if r != 1:
                    print('-----------------------------------------------------------------------------------------------')
                if rank == 0:
                    sum_rewards += r
                    if args.render:
                        env.render()

                observations.append(o)
                actions.append(a)
                values.append(v)
                rewards.append(r)
                probs.append(p)
                if done:
                    o = env.reset()
                    #self.total_reward = 0
                    if rank == 0:
                        print('----------------------------------')
                        print('total reward of the episode:', sum_rewards)
                        print('----------------------------------')
                        if args.save_mode == 'all':
                            torch.save(agent.local_brain, os.path.join(args.log_dir, args.save_name+"_{}.pkl".format(global_t[0])))
                        elif args.save_mode == 'last':
                            torch.save(agent.local_brain, os.path.join(args.log_dir, args.save_name+'.pkl'))
                        elif args.save_mode == 'max':
                            if max_sum_rewards < sum_rewards:
                                torch.save(agent.local_brain, os.path.join(args.log_dir, args.save_name+'.pkl'))
                                max_sum_rewards = sum_rewards
                        step = 0
                    break
                else:
                    #self.total_reward += r
                    _, v = agent.local_brain(torch.from_numpy(o).unsqueeze(0).float())
                    R += v.data.squeeze().item()

            # -- Agent advantage_push_agent.local_brain() --- 割引報酬和の計算
            
            returns = []
            for r in rewards[::-1]: # 割引報酬和
                R = r + 0.99 * R
                returns.insert(0, R)
            returns = torch.Tensor(returns)


            #if len(returns) > 1:
            #    returns = (returns-returns.mean()) / (returns.std()+args.eps)

            # -- LocalBrain _build_graph() --- lossの計算

            loss, v_loss, entropy, p_loss_list = agent._loss_function(actions, values, probs, returns, args)

            vs.append(v_loss.data.numpy())
            entropies.append(entropy.data.numpy())

            self.global_history_reward.append([tmp, sum_rewards])

            ## 記録
            if rank == 0 and done:
                logger.record_tabular_misc_stat('Entropy', entropies)
                logger.record_tabular_misc_stat('V', vs)
                logger.record_tabular('reward', sum_rewards)
                logger.record_tabular('step', global_t[0])
                logger.dump_tabular()
                del vs[:]
                del entropies[:]
                sum_rewards = 0

            # 重みの更新(最後まで)
            optimizer.zero_grad()
            final_node = [loss] + p_loss_list
            #print('final_node',final_node)
            gradients = [torch.ones(1)] + [None] * len(p_loss_list)
            #print('gradients',gradients)
            autograd.backward(final_node, gradients)
            #print('after_final_node',final_node)
            #print('after_gradients',gradients)

            #raise
            # 学習率の更新
            new_lr = np.true_divide(args.epoch - global_t[0] , args.epoch * args.lr)
            optimizer.step(new_lr)

            # cnt += 1

        send_rev.send(self.global_history_reward)
#'''

'''
    def train(self,rank, global_t, args):
        o = self.env.reset()
        step = 0
        sum_rewards = 0
        max_sum_rewards = 0
        vs = []
        entropies = []
        sum_rewards = 0
        cnt = 0
        while global_t[0] < args.epoch:
            #print('cnt:',cnt)
            agent.local_brain.model.sync(self.global_brain) # local policy にコピー
            observations = []
            actions = []
            values = []
            rewards = []
            probs = []
            R = 0
            for _ in range(args.local_t_max):
                global_t += 1
                step += 1
                # Agentのactで行動を取得
                #print('Variable(torch.from_numpy(o).float()).unsqueeze(0)',Variable(torch.from_numpy(o).float()).unsqueeze(0))
                p, v = agent.local_brain.model(Variable(torch.from_numpy(o).float()).unsqueeze(0))
                #p = torch.t(p)
                #print('p, v',p, v)
                #raise
                a = self.agent.act(o)
                if len(a.data.squeeze().size()) == 0:
                    o, r, done, _ = self.env.step(a.data.squeeze().item())
                else:
                    o, r, done, _ = self.env.step(a.data.squeeze()[0])
                print('r',r)
                if rank == 0:
                    sum_rewards += r
                    if args.render:
                        self.env.render()

                observations.append(o)
                actions.append(a)
                values.append(v)
                rewards.append(r)
                probs.append(p)
                if done:
                    o = self.env.reset()
                    if rank == 0:
                        print('----------------------------------')
                        print('total reward of the episode:', sum_rewards)
                        print('----------------------------------')
                        if args.save_mode == 'all':
                            torch.save(agent.local_brain.model, os.path.join(args.log_dir, args.save_name+"_{}.pkl".format(global_t[0])))
                        elif args.save_mode == 'last':
                            torch.save(agent.local_brain.model, os.path.join(args.log_dir, args.save_name+'.pkl'))
                        elif args.save_mode == 'max':
                            if max_sum_rewards < sum_rewards:
                                torch.save(agent.local_brain.model, os.path.join(args.log_dir, args.save_name+'.pkl'))
                                max_sum_rewards = sum_rewards
                        step = 0
                    break
                else:
                    _, v = agent.local_brain.model(torch.from_numpy(o).unsqueeze(0).float())
                    R += v.data.squeeze().item()

            # -- Agent advantage_push_agent.local_brain() --- 割引報酬和の計算
            returns = []
            for r in rewards[::-1]: # 割引報酬和
                R = r + 0.99 * R
                returns.insert(0, R)
            returns = torch.Tensor(returns)


            #if len(returns) > 1:
            #    returns = (returns-returns.mean()) / (returns.std()+args.eps)

            # -- LocalBrain _build_graph() --- lossの計算

            loss, v_loss, entropy, p_loss_list = self.agent._loss_function(actions, values, probs, returns, args)

            vs.append(v_loss.data.numpy())
            entropies.append(entropy.data.numpy())

            ## 記録
            if rank == 0 and done:
                logger.record_tabular_misc_stat('Entropy', entropies)
                logger.record_tabular_misc_stat('V', vs)
                logger.record_tabular('reward', sum_rewards)
                logger.record_tabular('step', global_t[0])
                logger.dump_tabular()
                del vs[:]
                del entropies[:]
                sum_rewards = 0

            # 重みの更新(最後まで)
            optimizer.zero_grad()
            final_node = [loss] + p_loss_list
            gradients = [torch.ones(1)] + [None] * len(p_loss_list)
            autograd.backward(final_node, gradients)
            # 学習率の更新
            new_lr = np.true_divide(args.epoch - global_t[0] , args.epoch * args.lr)
            optimizer.step(new_lr)

            cnt += 1
'''
