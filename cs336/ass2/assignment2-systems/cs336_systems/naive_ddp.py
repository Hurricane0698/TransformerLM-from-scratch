import torch.distributed as dist
import torch.nn as nn
import torch


class NaiveDDP(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        # 遍历module参数并修改
        with torch.no_grad():  # 原地修改处于autograd开始状态的叶子tensor会被pytorch拒绝
            for parameter in self.module.parameters():
                dist.broadcast(parameter, src=0, async_op=False)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def synchronize_gradients(self):
        for parameter in self.module.parameters():
            if parameter.grad is None:  # 测试里有requires_grad=False
                continue
            dist.all_reduce(parameter.grad, async_op=False)
            parameter.grad.div_(dist.get_world_size())
