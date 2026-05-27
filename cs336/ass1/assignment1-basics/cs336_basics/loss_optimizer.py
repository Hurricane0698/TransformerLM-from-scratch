import torch
import torch.nn as nn
import math
from typing import Any, Optional
from collections.abc import Iterable, Callable
from jaxtyping import Bool, Float, Int
from einops import rearrange, reduce, einsum
from torch import Tensor
def cross_entropy(inputs: Float[Tensor, " ... vocab_size"], targets: Int[Tensor, "..."])->Float[Tensor, " "]:
    #从input里vocab维度选出最大的值，然后broadcasting减去这个值。然后取指数求和，最后用input/targets
    inputs = inputs.flatten(0, -2)
    targets = targets.flatten()
    max_logits = reduce(inputs, 'N vocab_size -> N 1', "max")
    stab_inputs = inputs - max_logits
    #arange device 默认cpu
    target_logits = stab_inputs[ torch.arange(targets.shape[0],device=inputs.device), targets].unsqueeze(1)#stab_input.gather(1, targets.unsqueeze(1))
    l_t = torch.log(reduce(torch.exp(stab_inputs), 'N vocab_size -> N 1', "sum")) - target_logits
    avg_l = reduce(l_t, 'N 1-> ', "mean")
    return avg_l

class AdamW(torch.optim.Optimizer):
    def __init__(self, 
                 params: Iterable[Tensor] | Iterable[dict[str, Any]] | Iterable[tuple[str, Tensor]], 
                 lr:float,
                 betas:tuple[float, float],
                 weight_decay:float,
                 eps:float):
        defaults = {"lr": lr, 
                    "betas": betas,
                    "eps": eps,
                    "weight_decay":weight_decay}
        super().__init__(params, defaults)
        '''最好不要写
        self.beta = beta
        self.lr = lr
        self.weight_decay = lamda
        self.sigma = sigma， 因为Optimizer需要支持多个参数groups，学习率、权重衰减都可能不一样，最好全部传给父类字典统一管理'''
    def step(self, closure: Optional[Callable] = None):
        #规范开头
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["betas"][0]
            beta2 = group["betas"][1]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                #检查状态，如果没有则初始化
                state = self.state[param]
                #m / v 初始化不能用empty_like，不是全零而是未定义垃圾值
                m = state.get("m", torch.zeros(param.shape, dtype=param.dtype, device=param.device))
                v = state.get("v", torch.zeros(param.shape, dtype=param.dtype, device=param.device))
                #开始更新，不变量t,grad,m,v,lr均要反映当前更新状态.学习率
                t = state.get("t", 1)
                grad = param.grad.data
                adjust_lr = lr*((1-beta2**t)**0.5/(1-beta1**t))
                param.data -= lr*weight_decay*param.data
                m = beta1*m + (1-beta1)*grad
                v = beta2*v + (1-beta2)*grad**2
                param.data -= adjust_lr*m/(v**0.5+eps)
                #事务
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        return loss

def learning_rate_schedule(t:int, 
                           max_learning_rate:float, 
                           min_learning_rate:float, 
                           T_w:int, 
                           T_c:int):
    if t < T_w:
        lr_t = (t / T_w) * max_learning_rate
    
    elif T_w <= t <= T_c:
        lr_t = min_learning_rate + 1/2*(max_learning_rate - min_learning_rate)*(1 + math.cos((t - T_w)/(T_c - T_w)*math.pi))
    
    else: 
        lr_t = min_learning_rate
    
    return lr_t

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    #先转为可以多次遍历的列表，传入iterator时可用
    parameters = list(parameters)
    total = 0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        part = reduce((parameter.grad)**2, "... -> ", "sum")#不绕过auto_grad修改data，因为这里没有梯度追踪
        total += part
    sqrt_total = total**0.5
    if sqrt_total > max_l2_norm:
        for parameter in parameters:
            if parameter.grad is None:
                continue
            parameter.grad.mul_((max_l2_norm / (sqrt_total + 1e-6)))#原地修改方法是后面加个_, grad有add,mul方法