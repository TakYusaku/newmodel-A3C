## task
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from torch.optim import Optimizer, RMSprop

class Policy(nn.Module):
    def __init__(self, num_actions, dim_obs=None, out_dim=16, frame_num=1):
        super(Policy, self).__init__()
        self.num_actions = num_actions
        self.dim_obs = dim_obs
        self.frame_num = frame_num
        self.out_dim = out_dim
        self.head = nn.Linear(dim_obs*frame_num, out_dim)
        self.p = nn.Linear(out_dim, num_actions)
        self.v = nn.Linear(out_dim, 1)

        self.train()

    def forward(self, x):
        x = F.relu(self.head(x))
        policy = self.p(x)
        value = self.v(x)
        return F.softmax(policy,dim=1), value

    def sync(self, global_module):
        for p, gp in zip(self.parameters(), global_module.parameters()):
            p.data = gp.data.clone()
