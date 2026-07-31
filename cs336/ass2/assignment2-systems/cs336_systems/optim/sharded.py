from typing import Any

import torch
import torch.distributed as dist


class ShardedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls: type[torch.optim.Optimizer], **kwargs: Any):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.optimizer_cls = optimizer_cls
        self._optimizer = None
        
        self._local_param_groups = []
        self._param_owners = {}
        self._owned_parameters = []

        # init会逐个调用add_param_group，实现同时记录完整group和本rank的切片
        super().__init__(params, kwargs)
        self._optimizer = optimizer_cls(self._local_param_groups, **kwargs)
        self._local_param_groups = self._optimizer.param_groups

        # 内层optimizer补全了AdamW等优化器自己的默认值，把它们映射回外层group供scheduler读取。
        self.defaults = self._optimizer.defaults
        for full_group, local_group in zip(self.param_groups, self._local_param_groups):
            for key, value in local_group.items():
                if key != "params":
                    full_group.setdefault(key, value)
        self.state = self._optimizer.state

    def step(self, closure=None, **kwargs):
        # scheduler修改的是外层group，step前将超参数传给一一对应的本地group
        self._sync_local_param_groups()
        loss = self._optimizer.step(closure, **kwargs)

        # 每个参数只在owner上更新后把新值同步给所有rank
        for parameter in self._owned_parameters:
            dist.broadcast(parameter.data, src=self._param_owners[parameter])
        return loss

    def add_param_group(self, param_group: dict[str, Any]):
        # 先让基类完成generator展开和默认超参数填充，再按全局参数顺序轮流分配owner
        super().add_param_group(param_group)
        full_group = self.param_groups[-1]
        local_parameters = []
        for parameter in full_group["params"]:
            if parameter not in self._param_owners:
                owner = len(self._owned_parameters) % self.world_size
                self._param_owners[parameter] = owner
                self._owned_parameters.append(parameter)
            if self._param_owners[parameter] == self.rank:
                local_parameters.append(parameter)

        # 保留原group的超参数边界，只把params替换成本rank拥有的部分
        local_group = {key: value for key, value in full_group.items() if key != "params"}
        local_group["params"] = local_parameters
        if self._optimizer is None:
            self._local_param_groups.append(local_group)
        else:
            self._optimizer.add_param_group(local_group)

    def load_state_dict(self, state_dict):
        # checkpoint按完整参数编号保存；恢复后内层optimizer继续引用本rank对应的state
        super().load_state_dict(state_dict)
        self._optimizer.state = self.state
        self._sync_local_param_groups()

    def _sync_local_param_groups(self):
        for full_group, local_group in zip(self.param_groups, self._local_param_groups):
            for key, value in full_group.items():
                if key != "params":
                    local_group[key] = value
