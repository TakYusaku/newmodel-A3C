## task
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from torch.optim import Optimizer, RMSprop

class AsyncRMSprop(RMSprop):
    def __init__(self, global_params, local_params, lr=1e-2, alpha=0.99, eps=0.1, weight_decay=0,centered=False):
        defaults = dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay,centered=False)
        super(AsyncRMSprop, self).__init__(global_params, **defaults)

        self.local_params_group = list(local_params)
        if not isinstance(self.local_params_group[0], dict):
            self.local_params_group = [{'params': self.local_params_group}]

        for l_group, group in zip(self.local_params_group, self.param_groups): # self.param_groups は global_paramsについて
            for l_p, p in zip(l_group['params'], group['params']):
                state = self.state[id(p)]
                # State initialization
                if len(state) == 0:
                    # grad = l_p.grad.data
                    l_pData = l_p.data
                    state['step'] = torch.IntTensor(1).share_memory_()
                    state['square_avg'] = l_pData.new().resize_as_(l_pData).zero_().share_memory_()

    def step(self, lr, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for l_group, group in zip(self.local_params_group, self.param_groups):
            for l_p, p in zip(l_group['params'], group['params']):
                if l_p.grad is None:
                    continue
                grad = l_p.grad.data
                state = self.state[id(p)]

                square_avg = state['square_avg']

                alpha = torch.tensor([group['alpha']],dtype=torch.float)

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                torch.addcmul(square_avg.mul_(alpha), grad, grad, value=1-alpha.item())

                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.mul_(alpha).add_(1 - alpha, grad)
                    avg = square_avg.addcmul(
                        -1, grad_avg, grad_avg).sqrt().add_(group['eps'])
                else:
                    avg = square_avg.sqrt().add_(group['eps'])

                # print('p.data',p.data)
                #p.data.addcdiv_(-lr, grad, avg)
                torch.addcmul(p.data, grad, avg, value=-lr)

        return loss

    def zero_grad(self):
        for l_group in self.local_params_group:
            for l_p in l_group['params']:
                if l_p.grad is not None:
                    l_p.grad.data.zero_()
                else: #elif l_p.grad is None:
                    continue