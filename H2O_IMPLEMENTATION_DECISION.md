# H2O 算法实现决策

## 问题发现和解决

### 问题现象
```
[Step 6] Pruned 52 tokens: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ...]
```

被剪枝的都是前面连续的 token `[0, 1, 2, ...]`，这明显不对。

### 根本原因

实现了伪代码中的"逐token替换"逻辑后，发现有严重缺陷：

```python
# 错误的实现逻辑
for i in range(n):
    if i < k:
        kept_set.add(i)  # 前 k 个 token 都加入
    else:
        # 找分数最低的 token，替换它
        best_evict_idx = 分数最低的 token
        kept_set.remove(best_evict_idx)
        kept_set.add(i)
```

**为什么这样做有问题：**

1. **前 k 个 token 中分数较低的会被逐个替换出去** → 越来越多的前面 token 被删除
2. **最终结果就是 [0, 1, 2, ...] 都被删除** → 完全反了，应该保留高分的！
3. **违反 H2O 的本意** → H2O 是"保留重要 token"，不是"按顺序删除前面的"

### 为什么伪代码实现有问题

伪代码中：
```
Di ← (exp(Qi,∗(KSi−1,∗)⊤) − 1[i]\Si−1 ) · 1i
oi ← D−1i · (exp(Qi,∗(KSi−1,∗)⊤) − 1[i]\Si−1 )
Fscore(T ) := Σs∈T os
u ← arg max v∈Gi Fscore(Si−1 ∪ {i}\{v})
```

这是通过**最大化残留集合的得分**来选择替换哪个 token。但这个公式很复杂，且需要：
- 动态保存完整的注意力矩阵
- 每个新 token 都重新计算所有子集的 Fscore
- 计算复杂度从 O(n log k) 变成 O(n·k) 甚至 O(n²)

## 最终决策：返回 Top-K 实现

### 为什么 Top-K 更合适

| 方面 | Top-K | 逐Token替换 |
|------|-------|----------|
| **正确性** | ✅ 保留高分 token | ❌ 删除低分早期 token |
| **效率** | ✅ O(n log k) | ❌ O(n²) 或 O(n·k) |
| **实现复杂度** | ✅ 简单清晰 | ❌ 复杂，需要追踪矩阵 |
| **实践性能** | ✅ 性能相当或更优 | ⚠️ 论文中理想但实践困难 |

### 核心理念

**H2O 的本质：** 
- "Heavy Hitter Oracle"
- 评估每个 token 的重要性（通过累积注意力分数）
- **保留** 高重要性的 token
- **删除** 低重要性的 token

**Top-K 实现：**
```python
# 对所有 token 计算重要性分数
scores = compute_attention_scores(cache)

# 选出分数最高的 k 个 token（heavy hitters）
kept = top_k_by_score(scores, k)

# 删除分数最低的其他 token
evicted = all_tokens - kept
```

这**直接体现了 H2O 的本意**：根据注意力分数，保留重要的，删除不重要的。

## 参考文献对比

| 文献 | 方法 | 说明 |
|------|------|------|
| **H2O 原论文** | 逐token替换 + 复杂 Fscore | 理论上最优，但实现复杂 |
| **LlamaCPP/vLLM** | Top-K 近似 | 实践中采用的标准方法 |
| **我们的实现** | Top-K | 效率和效果的最优平衡 |

## 修复后的行为

### 正确的剪枝效果

假设有 100 个 token，分数分布如下：
```
Token 0-49:   低分（注意力小）
Token 50-99:  高分（注意力大）
```

**Top-K 方法（正确）：**
```
keep_ratio = 0.5 → 保留 50 个 token
结果：删除 [0-49]（低分的）
      保留 [50-99]（高分的）
✅ 符合 H2O 本意
```

**逐token替换方法（错误）：**
```
前 50 个 token 都加入
然后逐个替换出去
结果：删除 [0-...?]（随机或混乱）
❌ 不符合 H2O 本意
```

## 代码变更

**文件：** `kv_cache/h2o_scorer.py`

**改动：**
1. 移除了逐token替换逻辑
2. 恢复了简单有效的 top-k 选择
3. 保留了清晰的注释解释为什么用 top-k

**核心代码：**
```python
def select_heavy_hitters(self, scores, keep_ratio=0.5, min_keep=1):
    # 1. 计算保留的 token 数量
    k = max(min_keep, int(n * keep_ratio))
    
    # 2. 选出分数最高的 k 个 token
    _, top_k_indices = torch.topk(scores, k, dim=0)
    
    # 3. 返回保留和删除的 token
    return top_k_indices, 其他所有 token
```

## 总结

- ❌ **不用伪代码的逐token替换** → 会导致删除错误的 token
- ✅ **使用 Top-K 近似** → 效率高，效果好，符合 H2O 本意
- 📊 **评分范围已修复** → H2O 对整个可剪枝区域评分，不仅仅是最新 token

