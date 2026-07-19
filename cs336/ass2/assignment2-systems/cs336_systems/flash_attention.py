from typing import Any

import torch
import math
import triton
import triton.language as tl
from torch import Tensor
# =============================================
# pytorch版实现，为triton版确定debug基础
# =============================================

class autogradFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q:Tensor, K:Tensor, V:Tensor, is_causal=False):
        #q,k,v [B,N,D] [4, 128, 64]
        B = Q.shape[0]
        N_q = Q.shape[1]
        D = Q.shape[2]
        D_v = V.shape[2]
        b_q = 16
        b_k = 32
        t_q = Q.shape[1] // b_q 
        t_kv = K.shape[1] // b_k
        #预先分配完整O和L,device相同
        O = V.new_empty((B, N_q, D_v))
        L = Q.new_empty((B, N_q))
        #分块计算，先取q再取kv计算同一个q分块产生的不同值最后相加
        for i in range(t_q):
            q_i = Q[:, i * b_q: i * b_q + b_q, :] #[B, b_q, D]
            o_i = torch.zeros(B, b_q, D_v)
            l_i = torch.zeros(B, b_q)
            m_i = Q.new_full((B, b_q), -torch.inf)
            for j in range(t_kv):
                k_j = K[:, j * b_k : j * b_k + b_k, :] #[B, b_k, D]
                v_j = V[:, j * b_k : j * b_k + b_k, :]
                s_ij = q_i @ k_j.transpose(1 , 2) / math.sqrt(D) #[B, b_q, b_k]
                tile_max = s_ij.amax(dim=-1)       # [B, b_q]
                m_new = torch.maximum(m_i, tile_max) #[B, b_q]
                P_ij = torch.exp(s_ij - m_new.unsqueeze(-1))#[B, b_q, b_k]            
                l_new = torch.exp(m_i - m_new) * l_i + torch.sum(P_ij, dim=-1, keepdim=False)
                C_ij = P_ij @ v_j
                alpha = torch.exp(m_i - m_new)
                rescaled_o = alpha.unsqueeze(-1) * o_i #(B,b_q,D_v)
                O_ij = rescaled_o + C_ij
                #更新l,m,o
                m_i = m_new
                l_i = l_new
                o_i = O_ij
            #写回总向量
            O_i = o_i / l_i.unsqueeze(-1)
            L_i = m_i + torch.log(l_i) #math.log 只接受单个 Python 标量
            q_start = i * b_q
            q_end = q_start + b_q
            O[:, q_start:q_end, :] = O_i
            L[:, q_start:q_end] = L_i
        #为backward保存
        ctx.save_for_backward(L, Q, K, V, O)
        return O

    @staticmethod
    def backward(ctx, x):
        raise NotImplementedError
    
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
        shape= (N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )
    v_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape= (N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )
    o_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape= (N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )
    l_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape= (N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )
  #复现flash attention算法，k,v block ptr需要用*_block_ptr.advance推进；o_i l m dtype为tl.float32, 计算的P需要转换到V的dtype,o也需要转换为合适dtype
  #投影方法：tensor.to， get pointer dtype方法：*_block_ptr.type.element_ty
  #实现：tl.load Q的tile, 创建初始o,l,m, for循环tl.cdiv k tile，每轮load相应k,v tile完成算法(过程中使用tl.dot做矩阵乘法)，online更新m,o,l,循环最后advance k,v ptr
  #其余部分原算法处理，(sqrtd)-1为输入scale
  #不变量：每轮k,v ptr对应计算值。
  #m_i为已扫描 key tiles 上每个 query row 的running maximum，o_i对应未归一化的value加权累加器，l_i代表在当前m_i的未归一化 softmax 分母
  #检查：shape,dtype
  #Q tile 和 m/l/O_i 活过整个 program；K/V/S/P 只活一轮 key 循环；最终只有 O/L 从片上状态逃逸到全局显存
    q_i = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")  #[b_q, D]
    l_i = tl.zeros((Q_TILE_SIZE, ), dtype = tl.float32)
    o_i = tl.zeros((Q_TILE_SIZE, D), dtype = tl.float32)
    m_i = tl.full((Q_TILE_SIZE,), value=-math.inf, dtype=tl.float32)

    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        k_j = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero") # [b_k, D]
        k_jt = tl.trans(k_j)
        v_j = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")
        s_ij = tl.dot(q_i, k_jt) * scale #[b_q, b_k]
        m_n = tl.maximum(m_i, tl.max(s_ij, axis = -1, return_indices = False))#[b_q, ]
        p_ij = tl.exp(s_ij - m_n[:, None])#[b_q, b_k]
        l_n = tl.exp((m_i - m_n)) * l_i + tl.sum(p_ij, axis = -1)
        acc = o_i * tl.exp(m_i - m_n)[:, None] #[b_q, D]
        o_ij = tl.dot(p_ij.to(v_j.dtype), v_j, acc=acc)

        m_i = m_n
        l_i = l_n
        o_i = o_ij
        #更新k, v block ptr offset
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))#advance 只接收一个 offsets 参数，这个参数本身是 tuple
        v_block_ptr = v_block_ptr.advance((K_TILE_SIZE, 0))
    O_i = o_i / l_i[:, None]
    L_i = m_i + tl.log(l_i)
    tl.store(o_block_ptr, O_i.to(o_block_ptr.type.element_ty),boundary_check=(0, 1))
    tl.store(l_block_ptr, L_i, boundary_check=(0,))

class MyTritonFlashAttentionAutogradFunctionClass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q:Tensor, K:Tensor, V:Tensor, is_causal=False):
        B = Q.shape[0]
        N_q = Q.shape[1]
        N_k = K.shape[1]
        D = Q.shape[-1]
        D_v = V.shape[-1]
        b_q = 16
        b_k = 32
        ctx.Q_TILE_NUM = triton.next_power_of_2(N_q) // b_q
        scale = 1/ math.sqrt(D)
        #预先分配完整O和L,device相同
        O = V.new_empty((B, N_q, D_v))
        L = Q.new_empty((B, N_q))
        flash_fwd_kernel[(ctx.Q_TILE_NUM, B)](
            Q,K,V,O,L,
            stride_qb=Q.stride(0),stride_qq=Q.stride(1),stride_qd=Q.stride(2),
            stride_kb=K.stride(0),stride_kk=K.stride(1),stride_kd=K.stride(2),
            stride_ob=O.stride(0), stride_oq=O.stride(1),stride_od=O.stride(2),
            stride_lb=L.stride(0),stride_lq=L.stride(1),stride_vb=V.stride(0),stride_vk=V.stride(1),stride_vd=V.stride(2),
            scale=scale, N_QUERIES=N_q, N_KEYS=N_k, D=D ,Q_TILE_SIZE=b_q ,K_TILE_SIZE=b_k
            )
        ctx.save_for_backward(L, Q, K, V, O)
        return O