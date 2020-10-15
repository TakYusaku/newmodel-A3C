## task
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from torch.optim import Optimizer

from parameterserver import Policy

# --各スレッドで走るTensorFlowのDeep Neural Networkのクラスです　-------
class LocalBrain:
    def __init__(self, num_actions, dim_obs=None, out_dim=512, frame_num=1):
        self.model = Policy(num_actions, dim_obs, out_dim, frame_num)



        '''
        # loss関数を定義します
        log_prob = tf.log(tf.reduce_sum(p * self.a_t, axis=1, keep_dims=True) + 1e-10)
        advantage = self.r_t - v
        loss_policy = - log_prob * tf.stop_gradient(advantage)  # stop_gradientでadvantageは定数として扱います
        loss_value = LOSS_V * tf.square(advantage)  # minimize value error
        entropy = LOSS_ENTROPY * tf.reduce_sum(p * tf.log(p + 1e-10), axis=1, keep_dims=True)  # maximize entropy (regularization)
        self.loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)
        '''

