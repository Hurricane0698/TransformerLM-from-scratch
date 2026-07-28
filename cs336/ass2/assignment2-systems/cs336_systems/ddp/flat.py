import torch
import torch.distributed as dist
import torch.nn as nn
import torch._utils


class FlatDDP(nn.Module):
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
        parameters_with_grad = []
        original_grads = []
        for parameter in self.module.parameters():
            if parameter.grad is None:  # 测试里有requires_grad=False
                continue
            # list保存tensor引用
            parameters_with_grad.append(parameter)
            original_grads.append(parameter.grad)
        # 发生tensor复制创建连续storageB
        flat_grad_buffer = torch._utils._flatten_dense_tensors(original_grads)
        dist.all_reduce(flat_grad_buffer, async_op=False)
        flat_grad_buffer.div_(dist.get_world_size())
        # 在storageB上创建多个view
        averaged_grad_views = torch._utils._unflatten_dense_tensors(flat_grad_buffer, original_grads)
        # 让 parameter.grad 直接引用平均后的 flat-buffer view，避免按梯度元素复制回原 storage
        for parameter, averaged_grad_view in zip(parameters_with_grad, averaged_grad_views, strict=True):
            parameter.grad = averaged_grad_view
