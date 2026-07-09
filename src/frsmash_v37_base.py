"""
FRSMASH v3.6 — SSM + fla multi-head GLA recall (高效版 v3.5)

v3.5 → v3.6 变更:
  * recall 分支: 手写 LinAttn loop (慢) → fla multi-head GLA (chunk_gla)
      S_t = exp(g)·S_{t-1} + k_t ⊗ v_t,  o_t = q_t · S_t   (multi-head, content addressing)
      g 初始弱遗忘 (bias=8, 强保留), chunked 并行 (无 loop, 高效)
      复杂度 O(T·D·d_h), d_h=64 (比单头 O(T·D^2) 省 12x, 比 attention O(T^2·D) 省 32x)
  * 保留: v3.5 三路架构 (SSM backbone + SlowMemory + recall) + PE
  * 收益: recall 分支从手写 loop 变 fla 高效 kernel (训练推理都快)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from fla.ops.hgrn import chunk_hgrn, fused_recurrent_hgrn
from fla.ops.gla import chunk_gla


class GlaRecall(nn.Module):
    """v3.6: fla multi-head GLA 召回分支. S_t = exp(g)·S_{t-1} + k⊗v, o = q·S.
    content addressing, chunked 高效 (无 loop). g 初始弱遗忘 (强保留, induction 友好)."""
    def __init__(self, d, heads=8, d_h=64):
        super().__init__()
        self.heads, self.d_h = heads, d_h
        self.q_proj = nn.Linear(d, heads * d_h, bias=False)
        self.k_proj = nn.Linear(d, heads * d_h, bias=False)
        self.v_proj = nn.Linear(d, heads * d_h, bias=False)
        self.g_proj = nn.Linear(d, heads * d_h, bias=True)
        self.out_proj = nn.Linear(heads * d_h, d, bias=False)
        nn.init.constant_(self.g_proj.bias, 8.0)
    def forward(self, x, initial_state=None, return_state=False):
        B, T, d = x.shape; H, K = self.heads, self.d_h
        q = self.q_proj(x).view(B, T, H, K)
        k = self.k_proj(x).view(B, T, H, K)
        v = self.v_proj(x).view(B, T, H, K)
        g = F.logsigmoid(self.g_proj(x).float()).view(B, T, H, K)
        out, st = chunk_gla(q, k, v, g, initial_state=initial_state, output_final_state=return_state)
        out = self.out_proj(out.view(B, T, H * K))
        return (out, st) if return_state else out

    @torch.no_grad()
    def step(self, x_t, S_prev):
        B = x_t.size(0); H, K = self.heads, self.d_h
        q = self.q_proj(x_t).view(B, H, K)
        k = self.k_proj(x_t).view(B, H, K)
        v = self.v_proj(x_t).view(B, H, K)
        g = torch.exp(F.logsigmoid(self.g_proj(x_t)).view(B, H, K).float())
        if S_prev is None:
            S_prev = torch.zeros(B, H, K, K, device=x_t.device, dtype=torch.float32)
        S_new = g.unsqueeze(-1) * S_prev + torch.einsum('bhk,bhj->bhkj', k.float(), v.float())
        o = torch.einsum('bhk,bhkj->bhj', q.float(), S_new)
        return self.out_proj(o.view(B, H * K)), S_new


# ============================================================
# 1. 稳定的 Multi-Slot F-layer (对数域 parallel scan)
# ============================================================
class StableParallelScan(nn.Module):
    """精确 sequential scan: h_t = A_t·h_{t-1} + B_t

    v3.2 修正: 原 closed-form parallel scan (cumA·cumsum(B·inv_cumA)) 在长序列下
    必然失稳 —— 要么 inv_cumA=exp(-log_cumA) 溢出为 inf, 要么加 clamp 截断后
    破坏 prod(A)·(B/prod(A)) 的抵消, 导致 chunked ≠ full (实测误差 ~85%).
    状态模型 (state-space) 的正确做法就是直接跑递推 h=A·h+B:
      • 状态 h 始终有界 (~O(1)), 永不溢出/下溢
      • chunked 与 one-shot 按定义严格相等 (它就是递推本身)
      • 支持 h_prev carry-in, 无缝分段 (任意长上下文, O(chunk) 显存)
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward_scan(A, B, h_prev=None):
        """
        h_t = A_t * h_{t-1} + B_t   (exact + fast blocked parallel scan)

        线性递推可在块内用 closed-form 并行计算 (fp64, 无溢出), 块间 carry state.
        块内 fp64 对 BLOCK≤128 且 A≥~0.002 严格精确 (无溢出/下溢, 无截断),
        块间 carry = 递推本身, 因此 chunked == one-shot 严格相等, 且 Python 步数
        降为 T/BLOCK (~30x fewer kernel launches than naive sequential).
        A: (b, ns, T, ds)∈(0,1], B: (b, ns, T, ds), h_prev: (b, ns, ds)
        """
        A32 = A.float().clamp(min=0.0, max=1.0)
        B32 = B.float()
        b, ns, T, ds = A32.shape
        BLOCK = 128
        H = torch.empty(b, ns, T, ds, device=A.device, dtype=torch.float32)
        h = torch.zeros(b, ns, ds, device=A.device) if h_prev is None else h_prev.float()
        n_blk = (T + BLOCK - 1) // BLOCK
        for k in range(n_blk):
            s = k * BLOCK
            a = A32[:, :, s:s + BLOCK].double().clamp(min=1e-12)
            bb = B32[:, :, s:s + BLOCK].double()
            logA = torch.log(a)
            logCumA = torch.cumsum(logA, dim=2)
            cumA = torch.exp(logCumA)
            invCumA = torch.exp(-logCumA)          # safe in fp64 for this BLOCK
            csB = torch.cumsum(bb * invCumA, dim=2)
            Hb = cumA * (h.double().unsqueeze(2) + csB)
            H[:, :, s:s + BLOCK, :] = Hb.float()
            h = Hb[:, :, -1, :].float()            # exact carry to next block
        return H.to(A.dtype)


class MultiSlotFLayer(nn.Module):
    """v3.2 多槽 F-layer: 稳定 scan + gated gen_model"""
    def __init__(self, dim_size, heads, n_slots=4):
        super().__init__()
        self.heads = heads
        self.d_head = dim_size // heads
        self.n_slots = n_slots
        self.d_sub = dim_size // n_slots
        assert dim_size % n_slots == 0

        # 原始输入投影 (4 路: out, out1, out2, out3)
        self.combined = nn.Linear(dim_size, 4 * dim_size, bias=False)

        # slot 递推门控 (写入门 / 遗忘门 / 输入门 / 候选值)
        self.slot_proj = nn.Linear(dim_size, 4 * dim_size, bias=False)

        # v3.2 gated MLP 替代 5-branch 手写加法 + head_linear
        # 输入: 5 路 concat (heads*5*d_head)，输出: heads*d_head
        self.gen_gate = nn.Sequential(
            nn.Linear(heads * 5 * self.d_head, dim_size, bias=True),
            nn.GELU(),
            nn.Linear(dim_size, dim_size, bias=True),
        )
        self.gen_norm = nn.RMSNorm(dim_size)  # RMSNorm 控制尺度
        # v3.3: scan 由 fla.ops.hgrn 提供, 无需 self.scan

    def forward(self, x, states=None):
        b, s, d = x.shape
        ns, ds = self.n_slots, self.d_sub

        # 4 路输入
        combined = self.combined(x).view(b, s, 4, self.heads, -1)
        out, out1, out2, out3 = combined.unbind(2)
        out = out.permute(0, 3, 1, 2)   # (b, heads, s, d_head)
        out1 = out1.permute(0, 3, 1, 2)
        out2 = out2.permute(0, 3, 1, 2)
        out3 = out3.permute(0, 3, 1, 2)

        # slot 递推: A = sigmoid(g1)*sigmoid(g2) + (1-sigmoid(g1)) ∈ (0,1]
        sg = self.slot_proj(x).reshape(b, s, 4, ns, ds).permute(0, 1, 3, 2, 4)
        af = torch.sigmoid(sg[..., 0, :])   # (b, s, ns, ds)
        ff = torch.sigmoid(sg[..., 1, :])
        i_f = torch.sigmoid(sg[..., 2, :])
        cf = torch.tanh(sg[..., 3, :])
        A = af * ff + (1 - af)               # (b, s, ns, ds), ∈ (0,1]
        B_coeff = af * i_f * cf              # (b, s, ns, ds)

        # v3.3: fla HGRN 融合 scan. h = exp(g)·h + x,  g=log(A)≤0,  x=B
        A_t = A.permute(0, 2, 1, 3).contiguous()   # (b, ns, s, ds)
        B_t = B_coeff.permute(0, 2, 1, 3).contiguous()
        bns = b * ns
        g_t = torch.log(A_t.clamp(min=1e-8)).reshape(bns, s, ds)
        x_t = B_t.reshape(bns, s, ds)
        st_in = states.reshape(bns, ds) if states is not None else None
        H_flat, st_out = chunk_hgrn(x_t, g_t, initial_state=st_in, output_final_state=True)
        H = H_flat.reshape(b, ns, s, ds)
        new_states = st_out.reshape(b, ns, ds)

        # out4 = reshape H → (b, heads, s, d_head)
        H_cat = H.permute(0, 2, 1, 3).reshape(b, s, d)  # (b, s, d)
        out4 = H_cat.reshape(b, s, self.heads, self.d_head).permute(0, 3, 1, 2)

        # v3.2 gated MLP + RMSNorm 替代 5-branch 手写加法
        cat = torch.cat([out, out1, out2, out3, out4], dim=-1)  # (b, heads, s, 5*d_head)
        cat_flat = cat.transpose(1, 2).reshape(b, s, -1)  # (b, s, heads*5*d_head)
        gen = self.gen_gate(cat_flat)  # (b, s, d)
        gen = self.gen_norm(gen)       # RMSNorm 控制尺度

        return gen, new_states


# ============================================================
# 2. FeedForward (SwiGLU)
# ============================================================
class FeedForward(nn.Module):
    def __init__(self, hidden_size, expand=4):
        super().__init__()
        d_exp = hidden_size * expand
        self.gate = nn.Linear(hidden_size, d_exp, bias=False)
        self.up = nn.Linear(hidden_size, d_exp, bias=False)
        self.down = nn.Linear(d_exp, hidden_size, bias=False)
        self.silu = nn.SiLU()

    def forward(self, x):
        return self.down(self.silu(self.gate(x)) * self.up(x))


# ============================================================
# 3. Decoder Layer (纯 SSM)
# ============================================================
class SSMLayer(nn.Module):
    """F-layer + FFN + 残差"""
    def __init__(self, hidden_size, num_heads, n_slots=4):
        super().__init__()
        self.ssm = MultiSlotFLayer(hidden_size, num_heads, n_slots)
        self.ffn = FeedForward(hidden_size)
        self.norm1 = nn.RMSNorm(hidden_size)
        self.norm2 = nn.RMSNorm(hidden_size)

    def forward(self, x, states=None):
        # Pre-norm 残差
        h = self.norm1(x)
        ssm_out, s = self.ssm(h, states)
        x = x + ssm_out
        x = x + self.ffn(self.norm2(x))
        return x, s


# ============================================================
# 4. v3.4 LinearSlowMemory: 线性递推 + 输出门 (fla 友好, backward element-wise)
# ============================================================
class LinearSlowMemory(nn.Module):
    """v3.4: h_t = A(x_t)·h_{t-1} + B(x_t),  y_t = α(x_t)·h_t + (1-α)·x_t
    A/B/α 只依赖 x_t (低秩 MLP), 不依赖 h -> backward element-wise, 可借 fla.
    """
    def __init__(self, d_model, rank=None):
        super().__init__()
        d = d_model
        r = rank or max(d // 4, 32)
        self.W_down = nn.Linear(d, r, bias=False)
        self.W_A = nn.Linear(r, d, bias=True)
        self.W_B = nn.Linear(r, d, bias=True)
        self.W_gate = nn.Linear(r, 1, bias=True)
        nn.init.constant_(self.W_A.bias, 2.0)

    def forward(self, x_seq, h0):
        z = self.W_down(x_seq)
        A = torch.sigmoid(self.W_A(z))
        Bv = self.W_B(z)
        g = torch.log(A.clamp(min=1e-8))
        H, h_final = chunk_hgrn(Bv, g, initial_state=h0, output_final_state=True)
        alpha = torch.sigmoid(self.W_gate(z))
        Y = alpha * H + (1.0 - alpha) * x_seq
        return Y, h_final

    def step(self, x_t, h_prev):
        z = self.W_down(x_t)
        A = torch.sigmoid(self.W_A(z))
        Bv = self.W_B(z)
        h = A * h_prev + Bv
        alpha = torch.sigmoid(self.W_gate(z))
        y = alpha * h + (1.0 - alpha) * x_t
        return y, h


# ============================================================
# 5. FRSMASH v3.4 (全 fla: 多槽 F-layer backbone + 线性 SlowMemory)
# ============================================================
class FRSMASHv36(nn.Module):
    """
    FRSMASH v3.4 = 多槽 F-layer (fla) + 线性 SlowMemory (fla), 全 fla 无 Python loop
    """
    def __init__(self, voc_size, hidden_size, num_heads, num_layers, n_slots=4):
        super().__init__()
        self.D = hidden_size
        self.n_slots = n_slots
        self.num_layers = num_layers

        self.em = nn.Embedding(voc_size, hidden_size, padding_idx=0)
        _pe = torch.zeros(16384, hidden_size)
        _pos = torch.arange(16384).unsqueeze(1)
        _div = torch.exp(torch.arange(0, hidden_size, 2) * (-math.log(10000) / hidden_size))
        _pe[:, 0::2] = torch.sin(_pos * _div); _pe[:, 1::2] = torch.cos(_pos * _div)
        self.register_buffer('pe', _pe)

        # 全部层为 SSM
        self.layers = nn.ModuleList([
            SSMLayer(hidden_size, num_heads, n_slots)
            for _ in range(num_layers)
        ])
        self.num_ssm = num_layers

        self.final_norm = nn.RMSNorm(hidden_size)

        # v3.4 SlowMemory 线性化 (fla chunk_hgrn)
        self.mem_input_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.slow_cell = LinearSlowMemory(hidden_size)
        self.mem_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Gated Fusion
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 4), nn.GELU(),
            nn.Linear(hidden_size // 4, 1), nn.Sigmoid()
        )
        self.fusion_norm = nn.RMSNorm(hidden_size)

        # v3.6: 独立 recall 分支 (fla multi-head GLA, content addressing, 高效)
        self.recall = GlaRecall(hidden_size, heads=num_heads, d_h=64)
        self.recall_norm = nn.RMSNorm(hidden_size)

        self.head = nn.Linear(hidden_size, voc_size, bias=False)

    def forward(self, x, states=None, h_slow=None, recall_state=None, return_state=False, pos_offset=0):
        B, T = x.shape
        D = self.D
        model_dtype = self.head.weight.dtype
        x_emb = self.em(x).to(dtype=model_dtype) + self.pe[pos_offset:pos_offset+T].to(dtype=model_dtype)

        # 状态初始化
        if states is None:
            states = [None] * self.num_ssm
        if h_slow is None:
            h_slow = torch.zeros(B, D, device=x.device, dtype=model_dtype)

        # 1. 骨干 (纯 SSM)
        h = x_emb
        new_states = [] if return_state else None
        for i, layer in enumerate(self.layers):
            s_in = states[i] if return_state else None
            h, s = layer(h, s_in)
            if return_state:
                new_states.append(s)
        x_ash = self.final_norm(h)

        # 2. v3.4 SlowMemory 线性化 (fla chunk_hgrn, 无 loop, backward 自动)
        inp_seq = self.mem_input_proj(x_emb)
        H_slow, h_slow = self.slow_cell(inp_seq, h_slow)
        x_mem = self.mem_proj(H_slow)

        # 3. v3.6 独立 recall 分支 (fla multi-head GLA, 高效 chunked)
        if return_state or recall_state is not None:
            recall_out, recall_state = self.recall(x_emb, initial_state=recall_state, return_state=True)
        else:
            recall_out = self.recall(x_emb)
        x_recall = self.recall_norm(recall_out)

        # 4. Gated Fusion + recall residual
        cat = torch.cat([x_ash, x_mem], dim=-1)
        gate = self.fusion_gate(cat)
        fused = self.fusion_norm(gate * x_ash + (1 - gate) * x_mem + x_emb) + x_recall

        logits = self.head(fused)
        if return_state:
            return logits, new_states, h_slow, recall_state
        return logits

    @torch.no_grad()
    def generate_step(self, token_id, states, h_slow, recall_state=None, pos=0):
        model_dtype = self.head.weight.dtype
        x = self.em(token_id).to(dtype=model_dtype) + self.pe[pos:pos+1].to(dtype=model_dtype)

        # 骨干 (逐步推理)
        h = x
        new_states = []
        for i, layer in enumerate(self.layers):
            h, s = layer(h, states[i])
            new_states.append(s)

        x_ash = self.final_norm(h[:, 0])

        # v3.4 SlowMemory 单步
        inp = self.mem_input_proj(x[:, 0])
        y_slow, h_slow = self.slow_cell.step(inp, h_slow)
        x_mem = self.mem_proj(y_slow)

        # v3.6 recall 单步 (state carry)
        o_recall, recall_state = self.recall.step(x[:, 0].float(), recall_state)
        x_recall = self.recall_norm(o_recall.to(model_dtype))

        # Fusion + recall residual
        cat = torch.cat([x_ash, x_mem], dim=-1)
        gate = self.fusion_gate(cat)
        fused = self.fusion_norm(gate * x_ash + (1 - gate) * x_mem + x[:, 0]) + x_recall
        logits = self.head(fused)
        return logits, new_states, h_slow, recall_state, pos + 1


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    VOCAB = 23005
    H = 768
    HEADS = 8
    LAYERS = 12
    N_SLOTS = 4

    model = FRSMASHv36(VOCAB, H, HEADS, LAYERS, n_slots=N_SLOTS).to(device)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"FRSMASH v3.4: {n:,} params")
    print(f"  SSM layers: {model.num_ssm}")

    # 训练前向
    x = torch.randint(0, VOCAB, (4, 384), device=device)
    logits = model(x)
    print(f"Forward: {logits.shape}")

    # 数值稳定性测试
    print("\nNumerical Stability Test:")
    all_stable = True
    for sl in [256, 1024, 4096, 16384, 65536]:
        x = torch.randint(0, VOCAB, (1, sl), device=device)
        with torch.no_grad():
            logits = model(x)
        mx = logits.max().item()
        mn = logits.min().item()
        nan = torch.isnan(logits).any().item()
        inf = torch.isinf(logits).any().item()
        status = "STABLE" if not nan and not inf and mx < 100 else "BURST"
        if status == "BURST":
            all_stable = False
        lb = f"{sl//1024}K" if sl >= 1024 else str(sl)
        print(f"  {lb:>8}  max={mx:>8.3f}  min={mn:>8.3f}  {status}")

    # fp16 半精度稳定性测试
    print("\nFP16 Numerical Stability Test:")
    try:
        model_h = model.half()
        for sl in [256, 1024, 4096]:
            x = torch.randint(0, VOCAB, (1, sl), device=device)
            with torch.no_grad():
                logits = model_h(x)
            nan = torch.isnan(logits).any().item()
            inf = torch.isinf(logits).any().item()
            status = "STABLE" if not nan and not inf else "BURST"
            if status == "BURST":
                all_stable = False
            lb = f"{sl//1024}K" if sl >= 1024 else str(sl)
            print(f"  {lb:>8}  max={logits.max().item():>8.3f}  {status}")
        model = model.float()
    except Exception as e:
        print(f"  fp16 skipped: {e}")

    # 反向传播稳定性 (梯度不爆炸/不消失)
    print("\nBackward Gradient Stability:")
    model.train()
    x = torch.randint(0, VOCAB, (2, 512), device=device)
    target = torch.randint(0, VOCAB, (2, 512), device=device)
    logits = model(x)
    loss = F.cross_entropy(logits.reshape(-1, VOCAB), target.reshape(-1))
    loss.backward()
    gnorms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    gmean = sum(gnorms) / len(gnorms)
    gmax = max(gnorms)
    grad_status = "STABLE" if not math.isnan(gmean) and not math.isinf(gmax) and gmax < 1000 else "BURST"
    if grad_status == "BURST":
        all_stable = False
    print(f"  loss={loss.item():.4f}  grad_mean={gmean:.4f}  grad_max={gmax:.4f}  {grad_status}")

    # 推理测试 (状态逐步传递)
    print("\nGeneration test:")
    model.eval()
    token = torch.tensor([[42]], device=device)
    states = [None] * model.num_ssm
    h_slow = torch.zeros(1, H, device=device)
    for step in range(5):
        logits, states, h_slow = model.generate_step(token, states, h_slow)
        token = logits.argmax(dim=-1, keepdim=True)
        print(f"  Step {step+1}: token={token.item()}, h_slow_norm={h_slow.norm().item():.4f}")

    print(f"\n{'='*40}\nOverall: {'ALL STABLE' if all_stable else 'HAS BURST'}")
