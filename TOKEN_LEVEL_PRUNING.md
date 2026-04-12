## ✅ Token-Level Pruning 实现完成

---

## 🎯 关键改动

### 1. **Token-Level Pruning 支持** (`models/QwenLLMWithKVCache.py`)

新增 `_decode_with_token_level_pruning()` 方法：
- 替代原来的 `model.generate()` 自动生成
- 手动实现自回归循环
- **在每个生成的 token 后立即检查是否需要 pruning**
- 如果需要 pruning，立即执行，防止 cache 溢出

```python
# 在 _decode_with_token_level_pruning() 中的核心逻辑：
for token_idx in range(max_new_tokens - 1):
    # 生成一个 token
    outputs = self.model(...)
    self.current_cache_len += 1
    
    # ← 每个 token 生成后立即 prune
    if self.kv_manager and self.kv_manager.should_prune():
        self._do_pruning(piggyback_attentions=None)
    
    # 继续下一个 token
```

### 2. **Prune 频率调整** (`run_all_wiki_experiments_v2.py`)

配置改动：
```python
kv_config = {
    "prune_every_n": 1,  # 从 2 改为 1，每一步都尝试 prune
    "keep_ratio": 0.5,   # 保留 50% 的 heavy hitters
    ...
}
```

### 3. **自动路由** (`models/QwenLLMWithKVCache.py`)

在 `generate_incremental()` 中添加逻辑：
```python
if self.kv_manager and self.kv_manager.pruning_mode == "h2o":
    # H2O 方法使用 token-level pruning
    response_text, generated_len = self._decode_with_token_level_pruning(...)
else:
    # 其他方法使用高效的 model.generate()
    response_text, generated_len = self._decode(...)
```

---

## 📊 预期行为

### 之前（每个 step 只 prune 一次）
```
[TOKEN TRACKING] Step 5:
  ✓ Prefilled tokens [2200:2217] (18 tokens)
  ✓ Generated tokens [2218:2263] (46 tokens)
  ✗ Pruned 1 tokens by H2O: [18]
    Cache: 2264 → 2266
```
❌ Cache 还在增长，pruning 效果不明显

### 之后（每个自回归 token 后都 prune）
```
[TOKEN TRACKING] Step 5:
  ✓ Prefilled tokens [2200:2217] (18 tokens)
  ✓ Generated tokens [2218:2263] (46 tokens)
  ✗ Pruned 46 tokens by H2O: [pruned during generation]
    Cache: 2264 → 2200 (保持稳定)
```
✅ Cache 长度保持稳定，有效压缩

---

## 🔧 工作流程

```
step N: generate_incremental(new_observation)
  ├─ [1] Prefill new_observation tokens
  ├─ [2] Check if H2O mode
  ├─ [3] If H2O → use _decode_with_token_level_pruning()
  │   ├─ For each token in autoregressive loop:
  │   │   ├─ Generate token t_i
  │   │   ├─ cache_len += 1
  │   │   ├─ Check if should_prune() → YES (prune_every_n=1)
  │   │   └─ Execute pruning immediately
  │   └─ Return generated text
  ├─ [4] Else → use _decode() with model.generate()
  └─ [5] Update token tracking with total pruned info
```

---

## 💡 关键特性

✅ **真正的流式 pruning** - 不等到 step 结束，每个 token 后立即 prune  
✅ **稳定的 cache 长度** - 通过频繁 pruning 维持 cache 在目标大小  
✅ **灵活的配置** - 只对 H2O 方法启用，其他方法保持高效  
✅ **完整的 token tracking** - 记录每个 pruning 事件及其影响  

---

## 🚀 使用

清理旧检查点后运行：
```bash
rm -f results/*/react_kv_h2o*checkpoint*.json
python run_all_wiki_experiments_v2.py --experiment react_kv_h2o
```

观察输出中的 `[TOKEN TRACKING]` 部分，应该看到：
- 频繁的 pruning 事件（几乎每个生成的 token 后都有一次）
- Cache 长度相对稳定（在一个范围内波动）
- 总压缩率 = Total pruned / (Final cache + Total pruned)

---

## ⚠️ 性能考虑

新的 token-level pruning 会带来性能开销：
- ❌ 不再使用 `model.generate()` 的批量优化
- ❌ Python 循环中的 forward pass 会更慢
- ✅ 但 cache 大幅减少，可能抵消开销

如果性能太差，可以考虑：
1. 降低 `num_score_layers` 从 3 改为 1
2. 使用更大的 `observation_window` 来减少 pruning 频率
3. 考虑回到 `prune_every_n=2` 或更大的值

---

## 📝 文件修改清单

- ✅ `models/QwenLLMWithKVCache.py`
  - 新增 `_decode_with_token_level_pruning()` 方法（~90 行）
  - 修改 `generate_incremental()` 中的路由逻辑
  
- ✅ `run_all_wiki_experiments_v2.py`
  - 改动配置：`prune_every_n: 2 → 1`
  
- ✅ `token_tracker.py`（已有）
  - 支持完整的 pruning 事件记录

---

现在系统已准备好进行真正的流式 token-level pruning！
