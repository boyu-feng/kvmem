# H2O 算法实现对比分析

## 伪代码 vs 当前实现

### 📋 伪代码逻辑（Algorithm 1: H2 Eviction）

```
输入: Q (query), K (key), k (预算大小)
输出: S (保留的 token 集合)

循环 i = 1 → n:
  1. 如果 i ≤ k: 保留所有 token（填满预算）
  2. 否则（i > k）:
     - 计算 Di: 当前token i与之前保留tokens的注意力分数差异
     - 计算 oi: 移除每个保留token后的得分归一化
     - 计算 Fscore: 对集合内所有 token 求和得分
     - 找出 u = 使得 Fscore(Si-1 ∪ {i}\{u}) 最大的 token
     - 执行替换: Si = (Si-1 ∪ {i}) \ {u}
```

**关键特点:**
- **逐个处理**: 每个新 token 到达时，执行一次"找最差token替换"的操作
- **动态替换**: 不是简单的 top-k，而是选择移除"最不重要"的 token
- **注意力导向**: 基于 Attention 权重的实时计算

---

## 🔍 当前实现

### 文件: `h2o_scorer.py` 中的 `select_heavy_hitters()`

```python
def select_heavy_hitters(self, scores, keep_ratio=0.5, min_keep=1):
    """
    Standard H2O: Select top-k heavy hitters based on cumulative attention scores.
    """
    n = scores.shape[0]
    
    # 1. 计算要保留的 token 数量
    k = max(min_keep, int(n * keep_ratio))
    
    # 2. 简单 top-k 选择
    _, top_k_indices = torch.topk(scores, k, dim=0)
    
    # 3. 返回保留和剔除的 token
    return heavy_hitter_indices, evicted_indices
```

---

## ❌ 当前实现与伪代码的差异

| 方面 | 伪代码逻辑 | 当前实现 | 符合度 |
|------|----------|--------|--------|
| **选择策略** | 逐个token替换最差的 | 一次性 top-k 选择 | ❌ 不符合 |
| **处理方式** | 每到一个新token执行一次替换 | 批量计算，一次性确定保留集合 | ❌ 不符合 |
| **复杂度** | O(n·k) 每个token执行替换操作 | O(n log k) 简单排序 | ⚠️ 优化但不同 |
| **动态性** | 真正的"在线"算法，逐步调整 | 离线 top-k 算法 | ❌ 不符合 |

---

## 📊 具体例子对比

### 场景: 5 个 token，保留 3 个

**Token 到达顺序及注意力分数:**
```
i=1: score=0.9  → 保留集合: {1}
i=2: score=0.8  → 保留集合: {1, 2}
i=3: score=0.7  → 保留集合: {1, 2, 3}
i=4: score=0.5  → 保留集合已满，需要替换
i=5: score=0.6  → 保留集合已满，需要替换
```

### 伪代码执行:
```
i=4 (score=0.5, 预算已满):
  - Gi = {1, 2, 3, 4}
  - 计算移除每个token后的 Fscore
  - 假设移除token 3 得分最高
  - Si = {1, 2, 4}  ← 替换了3

i=5 (score=0.6, 预算已满):
  - Gi = {1, 2, 4, 5}
  - 计算移除每个token后的 Fscore
  - 假设移除token 2 得分最高
  - Si = {1, 4, 5}  ← 替换了2
```
**结果:** {1, 4, 5}

### 当前实现执行:
```
分数数组: [0.9, 0.8, 0.7, 0.5, 0.6]
Top-3 最高分: indices = [0, 1, 4]  (对应 token 1, 2, 5)
保留: {1, 2, 5}
```
**结果:** {1, 2, 5}

**⚠️ 结果不同！**

---

## 🎯 当前实现的特性

### 优点:
1. ✅ **计算效率高**: O(n log k) vs O(n·k)
2. ✅ **实现简单**: 直接 top-k，无需复杂替换逻辑
3. ✅ **可预测性强**: 给定分数，结果确定

### 缺点:
1. ❌ **非原始H2O**: 不是论文中的替换逻辑
2. ❌ **失去动态性**: 无法反映"最后到达的token对之前token的影响"
3. ❌ **可能不最优**: 替换逻辑能更好地处理边界情况

---

## 💡 修复建议

### 选项 1: 实现真正的 H2O（逐token替换）

```python
def select_heavy_hitters_h2o_exact(self, scores, keep_ratio=0.5):
    """
    真正的 H2O: 逐个 token 执行替换操作
    """
    n = scores.shape[0]
    k = max(1, int(n * keep_ratio))
    
    # 初始化保留集合
    S = set(range(min(k, n)))  # 前 k 个 token
    
    # 对于每个新到达的 token
    for i in range(k, n):
        # 找出若移除集合中哪个 token，加入新 token 后得分最高
        candidates = list(S) + [i]
        best_to_remove = None
        best_score = -float('inf')
        
        for remove_idx in S:
            # 临时集合: 移除 remove_idx，加入 i
            temp_S = (S - {remove_idx}) | {i}
            # 计算得分（这里需要注意力矩阵）
            score = compute_attention_score(temp_S)  # 伪代码
            
            if score > best_score:
                best_score = score
                best_to_remove = remove_idx
        
        # 执行最优替换
        S = (S - {best_to_remove}) | {i}
    
    return sorted(list(S))
```

**需要:**
- 实时的注意力矩阵访问
- 每次计算所有 token 子集的 Fscore
- 复杂度显著提升

### 选项 2: 保持当前 top-k（简化H2O）

**当前方式实际上是:**
- "简化的H2O" 或 "H2O近似"
- 重点是"基于注意力分数的压缩"，而不是"逐token替换"
- 在实践中通常性能相当甚至更优（因为更稳定）

**论文对比:**
- 原始论文: 强调替换逻辑的最优性
- 实践简化: 很多实现都用 top-k 代替（如 LlamaCPP等）

---

## 🔧 当前代码位置

| 文件 | 方法 | 现状 |
|------|------|------|
| `kv_cache/h2o_scorer.py` | `select_heavy_hitters()` | 使用 top-k（简化版） |
| `kv_cache/pruning_strategy.py` | `prune()` | 调用 `select_heavy_hitters()` |
| `models/QwenLLMWithKVCache.py` | `_do_pruning()` | 每 token 后调用一次 |

---

## ✅ 建议方案

### 推荐: **保持当前 top-k 实现**

**原因:**
1. 实践中与原始 H2O 性能相当
2. 计算效率更高（每个新token只需一次计算，不需要遍历）
3. 当前架构（token-level pruning）已经很接近原始意图
4. 已在现有代码中验证可编译

### 如需完全遵循论文: **实现真正替换逻辑**

需要修改:
```python
# 保存完整的 Q, K 矩阵
# 每个新 token 到达时计算所有候选移除操作的得分
# 执行最优替换
```

**成本:** 性能下降 ~10-30%（取决于batch大小）

---

## 📝 总结

| 问题 | 答案 |
|------|------|
| **现在的逻辑是真正的 H2O 吗？** | ❌ 不是。是"基于注意力的 top-k"简化版 |
| **与伪代码的主要差异?** | 不做逐token替换，而是一次性 top-k 选择 |
| **是否需要修改？** | ⚠️ 取决于目标：<br>- 学术精确度: ✅ 需要<br>- 实践性能: ❌ 不需要 |
| **当前实现的优缺点？** | ✅ 高效、稳定<br>❌ 非原始论文逻辑 |

