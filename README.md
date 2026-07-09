# FRSMASH v3.7

**FRSMASH v3.7 = v3.6 + DirectAdd fusion (去 gate, 直接相加)**

## v3.6 → v3.7 核心改动

v3.6 的 Gated Fusion:
```python
gate = sigmoid(MLP([x_ash, x_mem]))         # gate 训练后永远=1.0
fused = norm(gate*x_ash + (1-gate)*x_mem + x_emb) + x_recall
```

v3.7 的 DirectAdd:
```python
x_mem = mem_norm(mem_proj(slow_cell(x_emb)))  # 加 RMSNorm 修幅度
fused = norm(x_ash + x_mem + x_emb) + x_recall  # 无 gate, 直接相加
```

改动量: 删除 fusion_gate (2行), 加 mem_norm (1行), 改 fusion 公式 (1行)。

## 为什么

v3.6 的 fusion gate 在训练中坍缩到 1.0 (logit~1635, sigmoid 饱和), 导致 SlowMemory 贡献 0%。
根因: SlowMemory 输出幅度 355× 于 SSM 骨干 (无归一化), gate 被迫→1 保护骨干信号。

v3.7 修复:
1. x_mem 加 RMSNorm → 幅度归一化
2. 删 gate → 强制所有路贡献, 防止坍缩
3. 短上下文(seq512): ≈v3.6 (SlowMemory 冗余, 加不加一样)
4. **长上下文(seq≥2048): SlowMemory 的慢衰减递归与骨干互补 → ppl 降低**

## 验证结果 (3 种子, minimind pretrain, 80M, d512/L8)

| seq | v3.6 (gate) | v3.7 (DirectAdd) | 提升 |
|-----|-------------|------------------|------|
| 512 | 37.18 ± 0.13 | 37.20 ± 0.12 | 持平(噪声) |
| 2048 | 103.01 ± 1.13 | 102.21 ± 1.23 | **-0.8%** |
| 4096 | 79.97 ± 1.53 | **78.89 ± 1.41** | **-1.4% (3/3 种子全赢)** |

**优势随上下文长度增长**: 512 持平 → 2048 赢 0.8% → 4096 赢 1.4%。

## 原理: 多尺度时间互补

```
SSM 骨干(多槽门控): 高频(近期细节), 快衰减
SlowMemory(线性递归): 低频(长程趋势), 慢衰减 (A≈1, EMA-like)

短序列: 高频≈低频 → 冗余 → gate=1 正确
长序列: 骨干丢弃远处 → SlowMemory 补上 → 互补 → DirectAdd 有用
```

## 文件
- `src/frsmash_v37_base.py` — v3.6 原版 (继承基类)
- `src/frsmash_directadd.py` — v3.7 DirectAdd 架构 (FRSMASHDirectAdd)
- `src/frsmash_v36_infer.py` — 推理模块
- `experiments/` — 训练/评测/消融脚本
- `data/` — 实验结果文档
- `PAPER.md` — 完整论文

## 环境
PyTorch 2.x + CUDA, `fla` (flash-linear-attention), `jieba`
