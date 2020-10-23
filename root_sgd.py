import torch

from torch.optim.optimizer import Optimizer, required


class Root(Optimizer):
    def __init__(self, params, lr=0.1, amplifier=0.01, theta=1, damping=1, option='I', eps=1e-8, weight_decay=0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr, weight_decay=weight_decay)
        self.it = 0
        super(Root, self).__init__(params, defaults)
        
    def update_buf(self, prev_optimizer):
        for group, prev_group in zip(self.param_groups, prev_optimizer.param_groups):
            for p, prev_p in zip(group['params'], prev_group['params']):
                d_p = p.grad.data
                param_state = self.state[p]
                if p.grad is None:
                    continue
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                else:
                    prev_d_p = prev_p.grad.data
                    buf = param_state['momentum_buffer']
                    buf.sub_(prev_d_p).mul_(1 - 1/self.it).add_(d_p)
                if group['weight_decay'] != 0:
                    buf.add_(group['weight_decay'], p.data)

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
                param_state = self.state[p]
                if p.grad is None:
                    continue
                buf = param_state['momentum_buffer']
                p.data.add_(buf, alpha=-group['lr'])
        self.it += 1
        return loss

