import math

import torch
import triton
import triton.language as tl
from torch import Tensor
from einops import rearrange, einsum

# =============================================
# pytorch版实现，为triton版确定debug基础
# =============================================
#普通backward函数，编译后交给autogradFunc调用得到结果最后返回
def backward_pytorch(Q:Tensor,K:Tensor,V:Tensor,O:Tensor,dO:Tensor,L:Tensor, is_causal = False):
    d = Q.shape[-1]
    scale = 1 / math.sqrt(d)
    S = einsum(Q, K,"b n_q d, b n_k d -> b n_q n_k") * scale
    if is_causal:
        n_q = Q.shape[1]
        n_k = K.shape[1]
        index_q = torch.arange(0, n_q, device=S.device)
        index_k = torch.arange(0, n_k, device=S.device)
        mask = index_q[:, None] >= index_k[None, :] #[n_q, n_k]
        S = torch.where(mask, S, -torch.inf)
    D = torch.sum(O * dO, dim=-1, keepdim=True)
    # L为了稳定性保存成fp32，P进入矩阵乘法前转回输入dtype，否则bf16 benchmark会出现dtype不匹配
    P = torch.exp(S - L[:, :, None]).to(V.dtype)#b, n_q, n_k
    dV = einsum(P,dO, "b n_q n_k, b n_q d -> b n_k d")
    dP = einsum(dO, V, "b n_q d, b n_k d -> b n_q n_k")
    dS = P * (dP - D)
    dQ = einsum(dS,K,"b n_q n_k, b n_k d -> b n_q d") * scale
    dK = einsum(dS, Q, "b n_q n_k,b n_q d -> b n_k d") * scale
    return dQ, dK, dV

#benchmark会连续切换N和D，dynamic=True避免每个shape都重新生成一套backward graph
compiled_backward = torch.compile(backward_pytorch, dynamic=True)

class autogradFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q: Tensor, K: Tensor, V: Tensor, is_causal: bool = False) -> Tensor:
        #q,k,v [B,N,D] [4, 128, 64]
        B, N_q, D = Q.shape
        D_v = V.shape[-1]
        b_q = 16
        b_k = 32
        t_q = N_q // b_q
        t_kv = K.shape[1] // b_k
        scale = 1 / math.sqrt(D)
        #预先分配完整O和L,device相同
        O = V.new_empty((B, N_q, D_v))
        L = Q.new_empty((B, N_q))
        #分块计算，先取q再取kv计算同一个q分块产生的不同值最后相加
        for i in range(t_q):
            q_start = i * b_q
            q_end = q_start + b_q
            q_i = Q[:, q_start:q_end, :] #[B, b_q, D]
            o_i = V.new_zeros((B, b_q, D_v))
            l_i = Q.new_zeros((B, b_q))
            m_i = Q.new_full((B, b_q), -torch.inf)

            # 循环不变量：m_i、l_i、o_i 共同表示当前已经扫描过的全部 key tiles，
            # 后续 tile 只需更新这三个状态，而不需要重新读取之前的 K/V。
            for j in range(t_kv):
                kv_start = j * b_k
                kv_end = kv_start + b_k
                k_j = K[:, kv_start:kv_end, :] #[B, b_k, D]
                v_j = V[:, kv_start:kv_end, :]
                s_ij = q_i @ k_j.transpose(1, 2) * scale #[B, b_q, b_k]
                tile_max = s_ij.amax(dim=-1)  # [B, b_q]
                m_new = torch.maximum(m_i, tile_max) #[B, b_q]
                alpha = torch.exp(m_i - m_new)
                p_ij = torch.exp(s_ij - m_new.unsqueeze(-1))#[B, b_q, b_k]
                l_new = alpha * l_i + torch.sum(p_ij, dim=-1)
                c_ij = p_ij @ v_j

                # alpha 把旧状态从旧最大值 m_i 的指数基准，重标定到新最大值 m_new；
                # 分母 l_i 与未归一化分子 o_i 必须使用同一个重标定因子。
                rescaled_o = alpha.unsqueeze(-1) * o_i #(B,b_q,D_v)
                o_new = rescaled_o + c_ij
                #更新l,m,o
                m_i = m_new
                l_i = l_new
                o_i = o_new
            #写回总向量
            output_tile = o_i / l_i.unsqueeze(-1)
            logsumexp_tile = m_i + torch.log(l_i) #math.log 只接受单个 Python 标量

            # O 是归一化后的 attention 输出；L 保存每个 query row 的 log-sum-exp，
            # 它是 forward 数值稳定性的结果，也会成为 backward 重建 softmax 的紧凑状态。
            O[:, q_start:q_end, :] = output_tile
            L[:, q_start:q_end] = logsumexp_tile
        #为backward保存
        ctx.save_for_backward(L, Q, K, V, O)
        return O

    @staticmethod
    def backward(ctx, dO:Tensor) -> tuple[Tensor, Tensor, Tensor, None]:
        L, Q, K, V, O = ctx.saved_tensors
        dQ, dK, dV = compiled_backward(Q, K, V, O, dO, L)
        return dQ, dK, dV, None


#=========================================
# Triton forward实现
#=========================================
'''
Python Host
    │
    │ kernel launch: grid=(Tq, B), num_warps=...
    ▼
GPU 创建 Tq×B 个 program/CTA
    │
    ▼
CTA 调度到某个 SM
    │
    ▼
CTA 内多个 warps 开始执行
    │
    ▼
每个 warp 的 32 threads 协同处理 tile
    │
    ▼
load Q，循环读取 K/V，更新 m/l/O
    │
    ▼
store O/L
    │
    ▼
CTA/program 结束，片上资源释放
'''

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # triton program由(q,b)唯一标识，一个 Triton program 处理一个 batch 的一个 query tile；一个 program 内由多个 GPU threads 协作
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    #分别定义q,k,v,l,o的block指针
    Q_block_ptr = tl.make_block_ptr(
        #q_ptr为显存地址，Q_block_ptr 是 tile 地址描述表示应该从哪里加载多大一块，q_i是当前program加载并参与计算的triton tile值
        #对于参与的q_i，数据类型是tl.tensor
        #tl.tensor和torch.tensor主要不同之处在于：
        # 1.tl.tensor必须在编译期间确定shape,dtype从而确定运算方式针对性优化显存访问，torch...是运行时量，不提供tile级别的细致运算，显存读写频繁效率低。
        # 2.tl.tensor是GPU kernel编程语言下的数据类型，受限范围窄而torch.tensor完整且动态通用。总之就是效率和通用性的trade-off
        # 思考：能否做AI-power的编译器，兼顾通用性和高效率？
        #Q在GPU上运算的生命周期：1.Python 创建 torch.Tensor Q 
        # 2.CUDA allocator 为 Q 分配 GPU 全局显存
        # 2.kernel读取Q
        # 3.kernel结束后Q依然存在
        # 指针可以类比成制作tile的蓝图包 
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    v_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    o_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    l_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
  #复现flash attention算法，k,v block ptr需要用*_block_ptr.advance推进；o_i l m dtype为tl.float32, 计算的P需要转换到V的dtype,o也需要转换为合适dtype
  #投影方法：tensor.to， get pointer dtype方法：*_block_ptr.type.element_ty
  #实现：tl.load Q的tile, 创建初始o,l,m, for循环tl.cdiv k tile，每轮load相应k,v tile完成算法(过程中使用tl.dot做矩阵乘法)，online更新m,o,l,循环最后advance k,v ptr
  #其余部分原算法处理，(sqrtd)-1为输入scale
  #不变量：每轮k,v ptr对应计算值。
  #m_i为已扫描 key tiles 上每个 query row 的running maximum，o_i对应未归一化的value加权累加器，l_i代表在当前m_i的未归一化 softmax 分母
  #检查：shape,dtype
  #Q tile 和 m/l/O_i 活过整个 program；K/V/S/P 只活一轮 key 循环；最终只有 O/L 从片上状态逃逸到全局显存
    # 这里的三个 online-softmax 状态都使用 FP32：输入 tile 可以是低精度，
    # 但跨 key tiles 的 max、指数和累加误差会持续传播，因此状态精度决定整体稳定性。
    q_i = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")  #[b_q, D]
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    o_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    m_i = tl.full((Q_TILE_SIZE,), value=-math.inf, dtype=tl.float32)
    if is_causal:
        # mask用的query序列,[b_q]
        index_q_local = tl.arange(0, Q_TILE_SIZE)
        index_q_global = index_q_local + query_tile_index * Q_TILE_SIZE


    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        k_j = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero") # [b_k, D]
        #尾key会污染softmax结果，所以要mask掉
        index_k_local = tl.arange(0, K_TILE_SIZE)
        index_k_global = index_k_local + j * K_TILE_SIZE
        final_k_mask = N_KEYS > index_k_global
        if is_causal:
            masked = index_q_global[:,None] >= index_k_global[None, :]#[b_q, b_k]
        k_jt = tl.trans(k_j)
        v_j = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")
        s_ij = tl.dot(q_i, k_jt) * scale #[b_q, b_k]
        s_ij = tl.where(final_k_mask, s_ij, -1e6)
        if is_causal:
            #掩码为False处换为小常数
            s_ij = tl.where(masked, s_ij, -1e6)
        m_new = tl.maximum(m_i, tl.max(s_ij, axis=-1, return_indices=False))#[b_q, ]
        alpha = tl.exp(m_i - m_new)
        p_ij = tl.exp(s_ij - m_new[:, None])#[b_q, b_k]
        l_new = alpha * l_i + tl.sum(p_ij, axis=-1)
        acc = o_i * alpha[:, None] #[b_q, D]

        # p_ij 转回 V 的输入精度，使矩阵乘法走低精度高吞吐路径；
        # acc 仍为 FP32，因此 tl.dot 的跨 tile 累加状态不会随输入精度一起降低。
        o_new = tl.dot(p_ij.to(v_j.dtype), v_j, acc=acc)

        m_i = m_new
        l_i = l_new
        o_i = o_new
        #更新k, v block ptr offset
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))#advance 只接收一个 offsets 参数，这个参数本身是 tuple
        v_block_ptr = v_block_ptr.advance((K_TILE_SIZE, 0))

    # 循环结束后才执行唯一一次归一化；此前 o_i 始终是未归一化分子。
    output_tile = o_i / l_i[:, None]
    logsumexp_tile = m_i + tl.log(l_i)
    tl.store(o_block_ptr, output_tile.to(o_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(l_block_ptr, logsumexp_tile, boundary_check=(0,))


def triton_flash_attention_forward(Q: Tensor, K: Tensor, V: Tensor, q_tile_size: int, k_tile_size: int, num_warps: int, is_causal: bool):
    #对象: Q/K/V [B,N,D]，输出O [B,N,D]和L [B,N]
    #tile和num_warps是kernel编译参数，benchmark只改这三个值，不改kernel主体
    B, N_q, D = Q.shape
    N_k = K.shape[1]
    O = V.new_empty((B, N_q, D))
    L = torch.empty((B, N_q), device=Q.device, dtype=torch.float32)
    query_tile_count = triton.cdiv(N_q, q_tile_size)

    flash_fwd_kernel[(query_tile_count, B)](
        Q, K, V, O, L,
        stride_qb=Q.stride(0), stride_qq=Q.stride(1), stride_qd=Q.stride(2),
        stride_kb=K.stride(0), stride_kk=K.stride(1), stride_kd=K.stride(2),
        stride_vb=V.stride(0), stride_vk=V.stride(1), stride_vd=V.stride(2),
        stride_ob=O.stride(0), stride_oq=O.stride(1), stride_od=O.stride(2),
        stride_lb=L.stride(0), stride_lq=L.stride(1),
        N_QUERIES=N_q, N_KEYS=N_k, scale=1 / math.sqrt(D),
        D=D, Q_TILE_SIZE=q_tile_size, K_TILE_SIZE=k_tile_size, is_causal=is_causal,
        num_warps=num_warps,
    )
    return O, L


class MyTritonFlashAttentionAutogradFunctionClass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q: Tensor, K: Tensor, V: Tensor, is_causal: bool = False) -> Tensor:
        #作业adapter仍然走固定tile；需要sweep时由benchmark自己的Function传tile
        output, logsumexp = triton_flash_attention_forward(Q, K, V, 16, 32, 4, is_causal)
        ctx.is_causal = is_causal
        ctx.save_for_backward(logsumexp, Q, K, V, output)
        return output

    @staticmethod
    def backward(ctx, dO: Tensor) -> tuple[Tensor, Tensor, Tensor, None]:
        logsumexp, Q, K, V, output = ctx.saved_tensors
        dQ, dK, dV = compiled_backward(Q, K, V, output, dO, logsumexp, ctx.is_causal)
        return dQ, dK, dV, None
