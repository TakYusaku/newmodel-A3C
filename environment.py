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

# --CartPoleを実行する環境 -------
class Environment:
    total_reward_vec = np.zeros(10)  # 総報酬を10試行分格納して、平均総報酬をもとめる
    count_trial_each_thread = 0     # 各環境の試行数

    def __init__(self, env, args):
        self.global_history_reward = []
        #self.total_reward = 0

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
