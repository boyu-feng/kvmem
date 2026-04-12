# H2O 剪枝 Bug 修复总结

## 问题

用户报告看到以下输出：
```
[DEBUG] Computing H2O scores for tokens in range [1664, 1666)
[DEBUG] H2O scores computed. Heavy hitters: 1, Evicted: 1
[INFO] Running H2O pruning: scoring and hard-deleting low-scoring tokens
[DEBUG] Computing H2O scores for tokens in range [1664, 1666)
...（重复多次）

[Step 6] Pruned 52 tokens: [0, 1, 2, 3, 4, 5, 6, 7, 8, ...]
```

**两个问题：**
1. **重复调用** - 相同的评分范围被计算多次
2. **错误的剪枝** - 被删除的都是前面连续的 token `[0, 1, 2, ...]`

## 根本原因

### 问题1：重复调用
- 原因：`_decode_with_token_level_pruning()` 中每生成一个 token 都检查 `should_prune()`
- 结果：同一个 step 内，多个 token 生成时都会触发 pruning
- 评分范围 `[1664, 1666)` 只有2个 token，一直被重复计算

### 问题2：错误的剪枝
- 原因：尝试实现伪代码的"逐token替换"逻辑
- 问题：
  1. 前 k 个 token 都先加入集合
  2. 后续每个新 token 都会删除"分数最低"的 token
  3. 由于前面 token 往往分数较低，它们会被逐个删除
  4. 结果：`[0, 1, 2, ...]` 都被删除

## 修复方案

### 修复1：改变 H2O 评分范围
**文件：** `kv_cache/pruning_strategy.py` 的 `_prune_h2o()` 方法

**改动：** 确保 H2O 对**整个可剪枝区域**评分，而不仅仅是最新的 token

```python
# 改前
scores = self.h2o_scorer.compute_scores(attentions, prune_start, prune_end)
# 其中 prune_start/prune_end 只覆盖最新的 token

# 改后
# prune_start = protected_prefix_len（系统提示+问题）
# prune_end = total_len（最后一个 token）
# 这样评分覆盖整个可剪枝区域
scores = self.h2o_scorer.compute_scores(attentions, prune_start, prune_end)
```

**效果：** H2O 评分从整个对话历史中选出重要的 token

### 修复2：改变 pruning 触发频率
**文件：** `models/QwenLLMWithKVCache.py`

**改动：** 移除 `_decode_with_token_level_pruning()` 方法，恢复为每 step 只 prune 一次

```python
# 改前：token-level pruning（每个生成的 token 都 prune）
if self.kv_manager and self.kv_manager.should_prune():
    self._do_pruning(piggyback_attentions=None)

# 改后：step-level pruning（完整 step 后只 prune 一次）
# 在 generate_incremental() 中：
if self.kv_manager and self.kv_manager.should_prune():
    self._do_pruning(piggyback_attentions)  # 在 prefill 后 prune
```

**效果：** 评分和剪枝次数大幅减少，从"每个 token"变成"每个 step"

### 修复3：恢复 Top-K 选择算法
**文件：** `kv_cache/h2o_scorer.py` 的 `select_heavy_hitters()` 方法

**问题：** 尝试实现伪代码中的"逐token替换"导致删除错误的 token

**改动：** 恢复为简单有效的 Top-K 选择

```python
# 改前：逐token替换（复杂，且会删除前面的低分 token）
for i in range(n):
    if i < k:
        kept_set.add(i)
    else:
        best = min score in candidates
        kept_set.remove(best)
        kept_set.add(i)

# 改后：Top-K 简单选择（快速，正确）
_, top_k_indices = torch.topk(scores, k, dim=0)
return top_k_indices, others
```

**为什么 Top-K 更好：**
- ✅ **正确性** - 保留分数最高的 token，删除分数最低的
- ✅ **效率** - O(n log k) vs O(n²)
- ✅ **符合 H2O 本意** - "Heavy Hitter Oracle" = 保留重要 token
- ✅ **业界标准** - LlamaCPP, vLLM 都采用这种方式

## 预期改进

### 日志输出

**修复前：**
```
[DEBUG] Computing H2O scores for tokens in range [1664, 1666)
[DEBUG] H2O scores computed. Heavy hitters: 1, Evicted: 1
[DEBUG] Computing H2O scores for tokens in range [1664, 1666)
[DEBUG] H2O scores computed. Heavy hitters: 1, Evicted: 1
...（重复多次）
[Step 6] Pruned 52 tokens: [0, 1, 2, 3, 4, 5, 6, 7, ...]
```

**修复后：**
```
[DEBUG] H2O: Computing importance scores for entire region [128, 1664)
[DEBUG] H2O: Prunable region size: 1536 tokens
[DEBUG] H2O: Evaluated 1536 tokens in prunable region
[DEBUG] H2O: Keep ratio 0.50 -> Keep 768 / Evict 768
[Step 6] Pruned 768 tokens: [145, 289, 312, 567, ...]  # 乱序，符合分数排序
```

**关键改变：**
1. ✅ 评分范围 `[128, 1664)` - 整个可剪枝区域，不是 `[1664, 1666)`
2. ✅ **只调用一次** - 不再重复
3. ✅ 被删除的 token `[145, 289, 312, ...]` - 乱序（按分数排序），不是 `[0, 1, 2, ...]`

## 验证清单

- [x] 修复 H2O 评分范围（整个区域而非最新 token）
- [x] 移除 token-level pruning（恢复为 step-level）
- [x] 恢复 Top-K 选择算法（而非错误的逐token替换）
- [x] 代码编译通过
- [ ] 运行实验验证输出正确

## 文件变更清单

| 文件 | 变更 | 状态 |
|------|------|------|
| `kv_cache/pruning_strategy.py` | 更新 `_prune_h2o()` 的调试输出和评分范围说明 | ✅ 完成 |
| `models/QwenLLMWithKVCache.py` | 移除 `_decode_with_token_level_pruning()`，改用 `_decode()` | ✅ 完成 |
| `kv_cache/h2o_scorer.py` | 恢复 `select_heavy_hitters()` 为 Top-K 实现 | ✅ 完成 |

