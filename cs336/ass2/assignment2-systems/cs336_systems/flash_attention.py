import math

import torch
import triton
import triton.language as tl
from torch import Tensor
from einops import einsum

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

#=======================================
# Triton backward实现
#=======================================

'''P / dS 的 tile 网格

              key j →
          ┌───────────────┐
query i ↓ │ 00 │ 01 │ 02 │  → 横向归约得到 dQ_i
          ├────┼────┼────┤
          │ 10 │ 11 │ 12 │
          ├────┼────┼────┤
          │ 20 │ 21 │ 22 │
          └───────────────┘
            ↓    ↓    ↓
          纵向归约得到 dK_j、dV_j
          
          因此分成两个kernel'''
@triton.jit
def flash_bwd_dkdv_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr,
    L_ptr, D_ptr, dK_ptr, dV_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_dob, stride_doq, stride_dod,
    stride_lb, stride_lq,
    stride_db, stride_dq,
    stride_dkb, stride_dkk, stride_dkd,
    stride_dvb, stride_dvk, stride_dvd,
    N_QUERIES, N_KEYS, scale,
    D_HEAD: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    #对象: 当前program固定一个key tile
    #输入key/value [b_k,D]，循环读取query/dO [b_q,D]
    #输出grad_key/grad_value [b_k,D]，所以当前program对这两个tile有唯一写权限，不需要atomic add
    key_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    key_start = key_tile_index * K_TILE_SIZE

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D_HEAD),
        strides=(stride_kk, stride_kd),
        offsets=(key_start, 0),
        block_shape=(K_TILE_SIZE, D_HEAD),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D_HEAD),
        strides=(stride_vk, stride_vd),
        offsets=(key_start, 0),
        block_shape=(K_TILE_SIZE, D_HEAD),
        order=(1, 0),
    )
    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_index * stride_dkb,
        shape=(N_KEYS, D_HEAD),
        strides=(stride_dkk, stride_dkd),
        offsets=(key_start, 0),
        block_shape=(K_TILE_SIZE, D_HEAD),
        order=(1, 0),
    )
    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_index * stride_dvb,
        shape=(N_KEYS, D_HEAD),
        strides=(stride_dvk, stride_dvd),
        offsets=(key_start, 0),
        block_shape=(K_TILE_SIZE, D_HEAD),
        order=(1, 0),
    )
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D_HEAD),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D_HEAD),
        order=(1, 0),
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D_HEAD),
        strides=(stride_doq, stride_dod),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D_HEAD),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    key = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
    value = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
    key_offsets = key_start + tl.arange(0, K_TILE_SIZE)
    grad_key = tl.zeros((K_TILE_SIZE, D_HEAD), dtype=tl.float32)
    grad_value = tl.zeros((K_TILE_SIZE, D_HEAD), dtype=tl.float32)

    #循环不变量:
    #1.key/value不移动，始终是当前program负责的[b_k,D]
    #2.grad_key/grad_value只包含已经扫描过的query tiles的贡献，并且一直使用fp32累加
    #3.Q/dO/L/D四个指针始终指向相同的query tile，每轮结束同步advance
    for query_tile_index in range(tl.cdiv(N_QUERIES, Q_TILE_SIZE)):
        query = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
        grad_output = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero")
        logsumexp = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
        delta = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero")
        query_offsets = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)

        valid_pairs = (
            (query_offsets[:, None] < N_QUERIES)
            & (key_offsets[None, :] < N_KEYS)
        )
        if is_causal:
            valid_pairs = valid_pairs & (query_offsets[:, None] >= key_offsets[None, :])

        #query @ key.T: [b_q,D]，[D,b_k] ->  [b_q,b_k]
        #P不从forward保存，通过exp(scores-L)重新计算，从而不读写完整[N_q,N_k]矩阵
        scores = tl.dot(query, tl.trans(key)) * scale
        probability = tl.where(
            valid_pairs,
            tl.exp(scores - logsumexp[:, None]),
            0.0,
        )
        grad_probability = tl.dot(grad_output, tl.trans(value))  #[b_q,D]，[D,b_k] -> [b_q,b_k]
        grad_score = probability * (grad_probability - delta[:, None])  #[b_q,b_k]

        #两个输出的归约方向相同，都是把当前query tile的贡献加到固定key tile:
        #grad_value += P.T @ dO: [b_k,b_q]@[b_q,D] -> [b_k,D]
        #grad_key += dS.T @ Q: [b_k,b_q]@[b_q,D] -> [b_k,D]
        #scale只属于S=QK.T/sqrt(D)对Q/K的链式求导，所以dK需要乘scale，dV不需要
        #tl.dot输入转回Q/K/V的dtype让矩阵乘法走Tensor Core，
        #但acc参数仍是fp32，不同query tiles的累加不退回bf16
        grad_value = tl.dot(
            tl.trans(probability.to(grad_output.dtype)),
            grad_output,
            acc=grad_value,
        )
        grad_key = tl.dot(
            tl.trans(grad_score.to(query.dtype)),
            query,
            acc=grad_key,
        )

        Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))
        dO_block_ptr = dO_block_ptr.advance((Q_TILE_SIZE, 0))
        L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE,))
        D_block_ptr = D_block_ptr.advance((Q_TILE_SIZE,))

    tl.store(
        dK_block_ptr,
        (grad_key * scale).to(dK_block_ptr.type.element_ty),
        boundary_check=(0, 1),
    )
    tl.store(
        dV_block_ptr,
        grad_value.to(dV_block_ptr.type.element_ty),
        boundary_check=(0, 1),
    )


@triton.jit
def flash_bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr,
    L_ptr, D_ptr, dQ_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_dob, stride_doq, stride_dod,
    stride_lb, stride_lq,
    stride_db, stride_dq,
    stride_dqb, stride_dqq, stride_dqd,
    N_QUERIES, N_KEYS, scale,
    D_HEAD: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    #对象: 当前program固定一个query tile
    #输入query/output/dO [b_q,D]，循环读取key/value [b_k,D]
    #输出D [b_q]和grad_query [b_q,D]，两者都只由当前program写一次
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    query_start = query_tile_index * Q_TILE_SIZE

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D_HEAD),
        strides=(stride_qq, stride_qd),
        offsets=(query_start, 0),
        block_shape=(Q_TILE_SIZE, D_HEAD),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D_HEAD),
        strides=(stride_oq, stride_od),
        offsets=(query_start, 0),
        block_shape=(Q_TILE_SIZE, D_HEAD),
        order=(1, 0),
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D_HEAD),
        strides=(stride_doq, stride_dod),
        offsets=(query_start, 0),
        block_shape=(Q_TILE_SIZE, D_HEAD),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_start,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(query_start,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + batch_index * stride_dqb,
        shape=(N_QUERIES, D_HEAD),
        strides=(stride_dqq, stride_dqd),
        offsets=(query_start, 0),
        block_shape=(Q_TILE_SIZE, D_HEAD),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D_HEAD),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D_HEAD),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D_HEAD),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D_HEAD),
        order=(1, 0),
    )

    query = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    output = tl.load(O_block_ptr, boundary_check=(0, 1), padding_option="zero")
    grad_output = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero")
    logsumexp = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
    query_offsets = query_start + tl.arange(0, Q_TILE_SIZE)
    grad_query = tl.zeros((Q_TILE_SIZE, D_HEAD), dtype=tl.float32)

    #D_i=sum_r O_ir*dO_ir，只沿最后一个D维归约，所以每个query row相互独立
    #dQ kernel本来就要读取dO，再多读一个O tile就能在本program内完成D，不需要第三个kernel
    #output/dO可能是bf16，但D会进入softmax backward的每个key tile，所以乘法和归约保留fp32
    delta = tl.sum(output.to(tl.float32) * grad_output.to(tl.float32), axis=1)  #[b_q]
    #这个store不是同一kernel内program之间的barrier；当前dQ计算直接使用寄存器里的delta
    #写到global memory的副本只交给下一个dK/dV kernel使用
    tl.store(D_block_ptr, delta, boundary_check=(0,))

    #生命周期:
    #query/grad_output/logsumexp/delta/grad_query活过整个key循环
    #key/value/scores/probability/grad_probability/grad_score每轮重新产生，使用后即可释放
    #P不从forward保存，而由scores和L就地重建，这是用重复计算换掉[N_q,N_k]显存读写
    for key_tile_index in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        key = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        value = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        key_offsets = key_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)

        valid_pairs = (
            (query_offsets[:, None] < N_QUERIES)
            & (key_offsets[None, :] < N_KEYS)
        )
        if is_causal:
            valid_pairs = valid_pairs & (query_offsets[:, None] >= key_offsets[None, :])

        #boundary_check只保证load/store不会越界，不能阻止padding位置进入softmax
        #因此还要用valid_pairs把尾key和causal上三角的P显式变成0
        scores = tl.dot(query, tl.trans(key)) * scale
        probability = tl.where(
            valid_pairs,
            tl.exp(scores - logsumexp[:, None]),
            0.0,
        )
        grad_probability = tl.dot(grad_output, tl.trans(value))
        grad_score = probability * (grad_probability - delta[:, None])

        #grad_score [b_q,b_k] @ key [b_k,D] -> 当前key tile对grad_query [b_q,D]的贡献
        #先在fp32 grad_query里累加完整key方向，循环结束再乘scale并转换回输入dtype
        grad_query = tl.dot(
            grad_score.to(key.dtype),
            key,
            acc=grad_query,
        )

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))  #K/V必须同步advance，否则会把K_j和V_{j+1}配错
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    tl.store(
        dQ_block_ptr,
        (grad_query * scale).to(dQ_block_ptr.type.element_ty),
        boundary_check=(0, 1),
    )


def triton_flash_attention_backward(
    Q: Tensor, K: Tensor, V: Tensor, output: Tensor, grad_output: Tensor, logsumexp: Tensor,
    is_causal: bool, q_tile_size: int = 16, k_tile_size: int = 32, num_warps: int = 4,
):
    """Launch the two pure-Triton stages of Algorithm 2; no PyTorch fallback."""
    B, N_q, D_head = Q.shape
    N_k = K.shape[1]
    expected_shape = (B, N_q, D_head)
    if output.shape != expected_shape or grad_output.shape != expected_shape:
        raise ValueError("O and dO must have shape [B, N_q, D]")
    if K.shape != V.shape or K.shape[0] != B or K.shape[2] != D_head:
        raise ValueError("K and V must have shape [B, N_k, D]")
    if logsumexp.shape != (B, N_q):
        raise ValueError("L must have shape [B, N_q]")
    if D_head not in (16, 32, 64, 128):
        raise ValueError("Triton backward requires D in {16, 32, 64, 128}")
    if not (Q.dtype == K.dtype == V.dtype == output.dtype == grad_output.dtype):
        raise TypeError("Q, K, V, O and dO must share one dtype")

    dQ = torch.empty_like(Q)
    dK = torch.empty_like(K)
    dV = torch.empty_like(V)
    delta = torch.empty((B, N_q), device=Q.device, dtype=torch.float32)
    scale = 1 / math.sqrt(D_head)

    #两个kernel默认进入同一条current CUDA stream，同stream launch严格按顺序执行:
    #1.dQ kernel先计算并写出完整D
    #2.dQ kernel全部完成后dK/dV kernel才开始，因此后者能看到完整D
    #保证GPU工作顺序但不同步CPU 只有显式torch.cuda.synchronize才会让host等待
    flash_bwd_dq_kernel[(triton.cdiv(N_q, q_tile_size), B)](
        Q, K, V, output, grad_output, logsumexp, delta, dQ,
        stride_qb=Q.stride(0), stride_qq=Q.stride(1), stride_qd=Q.stride(2),
        stride_kb=K.stride(0), stride_kk=K.stride(1), stride_kd=K.stride(2),
        stride_vb=V.stride(0), stride_vk=V.stride(1), stride_vd=V.stride(2),
        stride_ob=output.stride(0), stride_oq=output.stride(1), stride_od=output.stride(2),
        stride_dob=grad_output.stride(0), stride_doq=grad_output.stride(1), stride_dod=grad_output.stride(2),
        stride_lb=logsumexp.stride(0), stride_lq=logsumexp.stride(1),
        stride_db=delta.stride(0), stride_dq=delta.stride(1),
        stride_dqb=dQ.stride(0), stride_dqq=dQ.stride(1), stride_dqd=dQ.stride(2),
        N_QUERIES=N_q, N_KEYS=N_k, scale=scale,
        D_HEAD=D_head, Q_TILE_SIZE=q_tile_size, K_TILE_SIZE=k_tile_size,
        is_causal=is_causal, num_warps=num_warps,
    )
    flash_bwd_dkdv_kernel[(triton.cdiv(N_k, k_tile_size), B)](
        Q, K, V, grad_output, logsumexp, delta, dK, dV,
        stride_qb=Q.stride(0), stride_qq=Q.stride(1), stride_qd=Q.stride(2),
        stride_kb=K.stride(0), stride_kk=K.stride(1), stride_kd=K.stride(2),
        stride_vb=V.stride(0), stride_vk=V.stride(1), stride_vd=V.stride(2),
        stride_dob=grad_output.stride(0), stride_doq=grad_output.stride(1), stride_dod=grad_output.stride(2),
        stride_lb=logsumexp.stride(0), stride_lq=logsumexp.stride(1),
        stride_db=delta.stride(0), stride_dq=delta.stride(1),
        stride_dkb=dK.stride(0), stride_dkk=dK.stride(1), stride_dkd=dK.stride(2),
        stride_dvb=dV.stride(0), stride_dvk=dV.stride(1), stride_dvd=dV.stride(2),
        N_QUERIES=N_q, N_KEYS=N_k, scale=scale,
        D_HEAD=D_head, Q_TILE_SIZE=q_tile_size, K_TILE_SIZE=k_tile_size,
        is_causal=is_causal, num_warps=num_warps,
    )
    return dQ, dK, dV

#=======================================
# Triton版autograd
#=======================================

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
        dQ, dK, dV = triton_flash_attention_backward(
            Q, K, V, output, dO, logsumexp, ctx.is_causal,
        )
        return dQ, dK, dV, None
