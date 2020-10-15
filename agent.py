## task
# EPS_START, EPS_END, EPS_STEPS, NUM_ACTIONS, frames, GAMMA, GAMMA_N, N_STEP_RETURN  をどこに突っ込むか
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from torch.optim import Optimizer
from torch.distributions.categorical import Categorical
import random
import numpy as np

from localbrain import LocalBrain
from parameterserver import Policy

frames = 0


GAMMA = 0.99
N_STEP_RETURN = 5
GAMMA_N = GAMMA ** N_STEP_RETURN

EPS_START = 0.5
EPS_END = 0.0
N_WORKERS = 8   # スレッドの数
EPS_STEPS = 200*N_WORKERS

# --行動を決定するクラスです、CartPoleであれば、棒付き台車そのものになります　-------
class Agent:
    def __init__(self, num_actions, dim_obs=None, out_dim=512, frame_num=1):
        self.NUM_ACTIONS = num_actions
        self.local_brain = Policy(num_actions, dim_obs, out_dim, frame_num)   # 行動を決定するための脳（ニューラルネットワーク）
        #self.memory = []        # s,a,r,s_の保存メモリ、　used for n_step return
        #self.R = 0.             # 時間割引した、「いまからNステップ分あとまで」の総報酬R
    
    def _loss_function(self,actions,values,probs,returns,args):
        p_loss = 0
        v_loss = 0
        entropy = 0
        p_loss_list = []
        for a, v, p, r in zip(actions, values, probs, returns):
            _p_loss = -self.m.log_prob(a) * (r - v.data.squeeze())
            p_loss_list.append(_p_loss)
            p_loss += p_loss
            # RSE of (v, r)
            #print('v, torch.Tensor([r])',v.shape, torch.Tensor([[r]]).shape)
            _v_loss = nn.MSELoss()(v, torch.Tensor([[r]]))
            v_loss += _v_loss
            # entropy
            entropy += -(p * (p + args.eps).log()).sum()
            
        v_loss = torch.tensor([v_loss],requires_grad = True) * 0.5 * args.v_loss_coeff
        entropy = entropy * args.entropy_beta
        loss = p_loss + v_loss - entropy

        loss, v_loss, entropy, p_loss_list
        #print('loss, v_loss, entropy, p_loss_list',loss, v_loss, entropy, p_loss_list)
        return loss, v_loss, entropy, p_loss_list

    def _loss_function_2(self, actions, values, probs, rewards, R, args):
        p_loss = 0
        v_loss = 0
        entropy = 0
        p_loss_list = []
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = rewards[i] + 0.99 * R
            advantage = R - torch.tensor([values[i].item()])
            #print('advantage',advantage)
            v_loss += 0.5 * args.v_loss_coeff * advantage.pow(2)

            #print('values[i + 1].data',values[i + 1].data)
            delta_t = rewards[i] + args.gamma * values[i + 1].data - values[i].data
            gae = gae * args.gamma * args.tau + delta_t

            entropy = -(probs[i] * (probs[i] + args.eps).log()).sum()
            p_loss -= ((probs[i].log()*Variable(gae).expand_as(probs[i].log())).sum() + (entropy * args.entropy_beta).sum())
            p_loss_list.append(p_loss)

        loss = p_loss + v_loss
        #print('loss',loss)
        #print('loss, v_loss, entropy, p_loss_list',loss, v_loss, entropy, p_loss_list)
        return loss, v_loss, entropy, p_loss_list


    # checked
    def act(self, s, eg_flg=False):# if eg_flg, do ε-greedy.
        if eg_flg: # ε-greedy法で行動を決定
            if frames >= EPS_STEPS:   # ε-greedy法で行動を決定します 171115修正
                eps = EPS_END
            else:
                eps = EPS_START + frames * (EPS_END - EPS_START) / EPS_STEPS  # linearly interpolate

            if random.random() < eps:
                return random.randint(0, self.NUM_ACTIONS - 1)   # ランダムに行動
            else:
                s = np.array([s])
                p = self.local_brain(s)

                # a = np.argmax(p)  # これだと確率最大の行動を、毎回選択

                a = np.random.choice(self.NUM_ACTIONS, p=p[0])
                # probability = p のこのコードだと、確率p[0]にしたがって、行動を選択
                # pにはいろいろな情報が入っていますが確率のベクトルは要素0番目
                return a
        else: # ε-greedy法を使わない
            p, _ = self.local_brain(torch.from_numpy(s).float().unsqueeze(0)) # p is policy, v is value 
            #print('p',p)
            self.m = Categorical(p)
            a = self.m.sample()
            #print('a',a)

            return a

