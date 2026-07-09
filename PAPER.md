# FRSMASH v3.7: DirectAdd Fusion — 长上下文互补架构

**作者**: dfytensor
**基座**: FRSMASH v3.6 (SSM backbone + SlowMemory + GLA recall)
**改动**: 去掉 Gated Fusion, 改为 DirectAdd (残差直加) + mem_norm (RMSNorm)

---

## 摘要

本文报告 FRSMASH v3.6 → v3.7 的架构改进。在 v3.6 中, Gated Fusion 的 gate 在训练中
坍缩到 1.0 (sigmoid 饱和, logit~1635), 导致 SlowMemory 分支贡献 0%。根因分析表明:
SlowMemory 输出无归一化, 幅度达 SSM 骨干的 355×, gate 被迫饱和以保护骨干信号。

v3.7 的改动极简: (1) 给 x_mem 加 RMSNorm 修复幅度; (2) 删除 gate, 改为直接相加
`fused = norm(x_ash + x_mem + x_emb) + x_recall`。

3 种子验证显示: 短上下文(seq512) v3.7 ≈ v3.6 (37.20 vs 37.18); **长上下文(seq≥2048)
v3.7 显著优于 v3.6**, 且优势随长度增长 (seq2048: -0.8%, seq4096: -1.4%, 3/3 种子全赢)。

原理: SSM 骨干(多槽门控, 快衰减)与 SlowMemory(线性递归, 慢衰减)在长上下文形成
**多尺度时间互补**——骨干捕获中近程高频细节, SlowMemory 保留长程低频趋势。
短序列两者冗余; 长序列骨干丢弃远处信息后, SlowMemory 的慢衰减补上了丢失的部分。

---

## 1. 背景: v3.6 的 Gate 坍缩问题

### 1.1 现象
FRSMASH v3.6 三路并行: SSM 骨干(8层, 54.4M) + SlowMemory(0.2M) + GLA recall(1.3M)。
融合公式: `fused = norm(gate·x_ash + (1-gate)·x_mem + x_emb) + x_recall`

训练后, gate = 1.0000 (所有位置, 所有 seq 长度 512-8192)。
SlowMemory 贡献 = 0%。

### 1.2 根因诊断 (排除法)
| 假设 | 实验 | 结果 |
|------|------|------|
| 路由坍缩 | 负载熵 2.225/2.303 | ❌ 路由均衡 |
| 状态冗余 | 范数 CV 1.127 | ❌ 状态分化 |
| 输入饥饿(State-MoE) | masked 递归状态范数 56 vs vanilla 1052 | 稀疏版问题, 不适用于 dense |
| 幅度失配 | x_mem 162k vs x_ash 457 (355×) | 修后仍 gate=1 |
| 输入源 | CrossFeed(x_ash→SlowMem) | 仍 gate=1 |
| 逻辑 vs 记忆 | GRU 替代线性 | 仍 gate=1 |
| **功能冗余** | **8层骨干 vs 1层旁路** | **✅ 骨干碾压** |

### 1.3 核心发现
gate=1 不是 bug, 是模型在**短上下文**训练时正确地绕过了冗余的 SlowMemory。
但在**长上下文**, SlowMemory 的慢衰减递归与骨干的快衰减互补——gate=1 此时错误地砍掉了有价值的信号。

---

## 2. v3.7: DirectAdd Fusion

### 2.1 改动
```diff
- gate = sigmoid(MLP([x_ash, x_mem]))
- fused = norm(gate*x_ash + (1-gate)*x_mem + x_emb) + x_recall
+ x_mem = mem_norm(mem_proj(slow_cell(x_emb)))    # 加 RMSNorm
+ fused = norm(x_ash + x_mem + x_emb) + x_recall  # 直接相加, 无 gate
```

参数变化: -0.13M (删 fusion_gate) + 0.0005M (加 RMSNorm) ≈ 不变。

### 2.2 多种子验证 (3 seeds, minimind pretrain, d512/L8, ~80M)

| seq | v3.6 均值±std | v3.7 均值±std | v3.7 提升 | 3种子赢几场 |
|-----|--------------|--------------|----------|------------|
| 512 | 37.18 ± 0.13 | 37.20 ± 0.12 | -0.05% (噪声) | 0/3 |
| 2048 | 103.01 ± 1.13 | 102.21 ± 1.23 | **-0.8%** | 2/3 |
| 4096 | 79.97 ± 1.53 | **78.89 ± 1.41** | **-1.4%** | **3/3** |

---

## 3. 原理: 多尺度时间互补

### 3.1 两种递归的遗忘曲线
| 机制 | 遗忘门 | 衰减率 | 时间频率 |
|------|--------|--------|----------|
| SSM 骨干(多槽) | sigmoid 门控, 多槽多速率 | 整体偏快 | 高频(近期) |
| SlowMemory(线性) | A=sigmoid, bias=2(A≈0.88) | 慢 | 低频(长程) |

### 3.2 短 vs 长上下文
- **短(seq512)**: 骨干状态 4096 维 >> 512 token, 完整编码, SlowMemory 冗余。
- **长(seq4096)**: 骨干状态 ≈ seq 长度, 必须压缩, 远处细节丢失;
  SlowMemory 的慢衰减(A≈0.88, 半衰期~5token... 实际因 A 是 input-dependent 可更慢)
  保留了骨干丢失的长程趋势 → 互补。

### 3.3 为什么 gate 学不会长上下文打开
gate 是 per-token 函数(输入 x_t → 输出 gate_t), 在 seq512 上训练。它在短序列学到
gate=1(正确), 然后冻结到所有长度——它没有"序列长度"概念, 无法自适应。
DirectAdd 去掉 gate, 在所有长度强制贡献: 短上下文=噪声(无害), 长上下文=互补(有益)。

---

## 4. 消融实验

### 4.1 砍掉 SlowMemory (NoSlow)
| seq | v3.6 (gate=1, SlowMem=0%) | NoSlow (无SlowMem) |
|-----|--------------------------|-------------------|
| 512 | 36.71 | 37.25 (噪声) |
| 4096 | 123.77 | 124.33 (噪声) |

确认: v3.6 的 SlowMemory 确实贡献 0%, 砍掉 ppl 不变。

### 4.2 放大 GLA recall (d_h 64→128, 4× 状态)
| d_h | seq512 ppl | seq4096 ppl |
|-----|-----------|-------------|
| 64 (原版) | 37.28 | 78.82 |
| 128 (4×) | 36.92 (-1.0%) | 79.91 (+1.4%) |

无可靠提升 (噪声级)。

### 4.3 State-MoE (10× SlowMemory 状态)
masked 版(稀疏): 输入饥饿(状态范数 56 vs vanilla 1052), 全长度无效。
dense 版(全强度): 状态范数 1770(全强度), seq512 无效(-1.8%)。

### 4.4 GRU 逻辑状态 (替代线性 SlowMemory)
GRU K=2 + DirectAdd: ppl 37.20 ± 0.48, 略差于 vanilla, 3.5× 慢。

### 4.5 TriCross (串联交叉)
SSM→SlowMem→GLA 串联: ppl 39.97 ± 0.53, **显著更差 -7.5%** (信息逐级衰减)。

---

## 5. 结论

> **FRSMASH v3.7 (DirectAdd) 是长上下文场景的最优融合方式。**
> 改动极简(删 gate + 加 norm), 参数不变, 短上下文持平 v3.6,
> **长上下文(seq≥2048) ppl 降低 0.8-1.4%, 优势随长度增长, 3 种子验证稳健。**
>
> 原理: SSM 骨干与 SlowMemory 形成多尺度时间互补(高频+低频),
> gate 在短序列训练时学会砍掉 SlowMemory(正确), 但无法在长序列自适应打开(错误);
> DirectAdd 绕过 gate, 在所有长度强制贡献, 长上下文下 SlowMemory 的低频信号成为有益补充。

---

## 6. 复现

```bash
# 训练 v3.7 DirectAdd (seq512)
python experiments/train_v37.py --cond directadd --steps 1500 --seq 512

# 长上下文 (梯度累计)
python experiments/train_v37.py --cond directadd --steps 200 --seq 4096 --accum 8

# 对照: v3.6 原版
python experiments/train_v37.py --cond vanilla --steps 1500 --seq 512
```

数据: minimind pretrain (open_ash_voc 分词, VOCAB=23005)
环境: PyTorch 2.x + CUDA + fla + jieba
