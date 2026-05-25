import torch
import numpy as np
import typing
import os
def data_loading(x:np.typing.NDArray, batch_size:int, context_length:int, device:str)->tuple[torch.Tensor, torch.Tensor]:
    #对象：x,batch_size, context_length, device , input, target, (input, target) i 状态变量：input, target, i
    #不变量：input每次更新后代表此次更新后最新的Tensor,target表示所有序列位置平移一位之后的Tensor.i表示当前抽样进度
    #首先验证x长度大于context_length+1（这里得考虑target而不是input）, 然后初始化i，循环随机确定起点，抽取context_length长度input, target然后更新两个tensor，i.
    #注意循环内要确认起点坐标加上context_length小于x的长度，否则继续且i不更新（更好的做法是先确定合法的范围，然后再在范围里抽取）
    #最后一次性把抽取完毕的input, target移动到同一个device上，然后拼成元组返回
    #上面是初版思路，实际代码为优化版
    assert len(x) >= (context_length + 1), "context length must smaller than length of data"
        #先找start
    starts = np.random.randint(low=0, high=len(x)-context_length, size=batch_size)
    offsets = np.arange(context_length)#上界是开区间
    #更新input, target,直接在batch_size一次性做完，不要一次一次创建，创建一个二维索引矩阵
    idx = starts[:, None] + offsets[None, :] #tensor[:, None].shape = [N, 1]。效果等价于unsqueeze
    inputs = torch.tensor(x[idx], device=device,dtype=torch.long)
    targets = torch.tensor(x[idx + 1], device=device, dtype=torch.long)
    return (inputs, targets)

def save_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,  
                    iteration: int,
                    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    saved_dict = {}
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()
    saved_dict["model"] = model_state
    saved_dict["optimizer"] = optimizer_state
    saved_dict["step"] = iteration
    torch.save(saved_dict, out)

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],  
                    model: torch.nn.Module,  
                    optimizer: torch.optim.Optimizer)-> int:
    saved_dict = torch.load(src)
    model.load_state_dict(saved_dict["model"])
    optimizer.load_state_dict(saved_dict["optimizer"])
    return saved_dict["step"]