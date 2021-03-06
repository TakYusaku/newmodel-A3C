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

"""Implements RMSprop algorithm with shared states.
"""
class SharedRMSprop(RMSprop):
    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False):
        super(SharedRMSprop, self).__init__(params, lr, alpha, eps, weight_decay, momentum, centered)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[id(p)]
                # State initialization
                if len(state) == 0:
                    state['step'] = torch.IntTensor(1).share_memory_()
                    state['square_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['square_avg'].share_memory_()
                    state['step'].share_memory_()
                    if group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['momentum_buffer'].share_memory_()
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['grad_avg'].share_memory_()

    def __setstate__(self, state):
        super(SharedRMSprop, self).__setstate__(state)            
                
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[id(p)]
                #print('state',state)

                square_avg = state['square_avg']
                alpha = group['alpha']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                torch.addcmul(square_avg.mul_(alpha), grad, grad, value=1-alpha)
                #square_avg.mul_(alpha).addcmul_(1 - alpha, grad, grad)

                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.mul_(alpha).add_(1 - alpha, grad)
                    avg = square_avg.addcmul(-1, grad_avg, grad_avg).sqrt().add_(group['eps'])
                else:
                    avg = square_avg.sqrt().add_(group['eps'])

                if group['momentum'] > 0:
                    buf = state['momentum_buffer']
                    buf.mul_(group['momentum']).addcdiv_(grad, avg)
                    p.data.add_(-group['lr'], buf)
                else:
                    '''
                    print('type(p)',p)
                    print("-group['lr']",-group['lr'])
                    print("avg",avg)
                    '''
                    torch.addcdiv(p.data, grad, avg, value=-group['lr'])
                    #p.data.addcdiv_(-group['lr'], grad, avg)

        return loss

class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()