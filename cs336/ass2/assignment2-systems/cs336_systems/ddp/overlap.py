import torch
import torch.distributed as dist
import torch.nn as nn


class OverlapDDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.handles = []
        for parameter in self.module.parameters():
            with torch.no_grad():
                dist.broadcast(parameter, src=0, async_op=False)
            if parameter.requires_grad is False:
                continue
            parameter.register_post_accumulate_grad_hook(self._callback)

    def _callback(self, parameter: torch.Tensor):
        # 先缩放再异步求和
        parameter.grad.div_(dist.get_world_size())
        handle = dist.all_reduce(parameter.grad, async_op=True)
        self.handles.append(handle)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()
