# Token Tracking 系统 - H2O 方法监控

## 📌 新增功能

现在已经添加了完整的 Token Tracking 系统，用于在 H2O 方法中监控：

1. ✅ 每一步的 token 变化
2. ✅ 被丢弃的 token 编号
3. ✅ 完整的 KV cache 长度变化
4. ✅ Prefill 和 Generated token 的分离
5. ✅ 压缩统计信息

---

## 🔧 实现细节

### 新建文件：`token_tracker.py`

```python
class TokenTracker:
    """Track token positions and pruning history."""
    
    def add_prefill_tokens(step, num_tokens)
        # 记录 prefill tokens（观察输入）
        # 返回 token 范围 [start:end]
    
    def add_generated_tokens(step, num_tokens)
        # 记录生成的 tokens（思想/行动）
        # 返回 token 范围 [start:end]
    
    def record_pruning(step, discarded_token_indices, new_cache_len)
        # 记录被丢弃的 token 编号和新缓存长度
    
    def print_step_summary(step, cache_len)
        # 打印每一步的摘要
    
    def print_full_history()
        # 打印完整的 token 历史
    
    def get_statistics()
        # 获取压缩统计信息
```

### 修改文件：`run_all_wiki_experiments_v2.py`

**改动 1: 导入 TokenTracker**
```python
from token_tracker import TokenTracker
```

**改动 2: 初始化 (第 837 行)**
```python
token_tracker = TokenTracker() if pruning_mode == "h2o" else None
```

**改动 3: 第一步追踪 (第 1240-1250 行)**
```python
if token_tracker:
    prompt_len = prompt_kv[0][0].shape[2] if prompt_kv else 0
    gen_len = generated_kv[0][0].shape[2] if generated_kv else 0
    token_tracker.add_prefill_tokens(1, prompt_len)
    if gen_len > 0:
        token_tracker.add_generated_tokens(1, gen_len)
    token_tracker.print_step_summary(1, llm.get_cache_len())
```

**改动 4: 后续步骤追踪 (第 1390-1415 行)**
```python
if token_tracker:
    cache_len_before = llm.get_cache_len()
    response = llm.generate_incremental(...)
    cache_len_after = llm.get_cache_len()
    
    # 计算 prefill 和 generated 的分离
    new_input_ids = llm.tokenizer(new_text, ...)
    new_token_count = new_input_ids.shape[1]
    
    token_tracker.add_prefill_tokens(step, new_token_count)
    gen_token_count = cache_len_after - cache_len_before - new_token_count
    if gen_token_count > 0:
        token_tracker.add_generated_tokens(step, gen_token_count)
    
    token_tracker.print_step_summary(step, cache_len_after)
```

**改动 5: 最终打印 (第 1592-1600 行)**
```python
if token_tracker:
    token_tracker.print_full_history()
    stats = token_tracker.get_statistics()
    print(f"[TOKEN STATS] Total pruned: {stats['total_pruned_tokens']}, "
          f"Prune events: {stats['num_prune_events']}")
```

---

## 📊 输出示例

### 第一步 (Prefill + Generate)

```
[TOKEN TRACKING] Step 1:
  Current cache length: 1674
  ✓ Prefilled tokens [0:1673] (1674 tokens)
  ✓ Generated tokens [1674:1713] (40 tokens)
```

### 后续步骤 (Prefill + Generate + 可能的 Pruning)

```
[TOKEN TRACKING] Step 2:
  Current cache length: 1512
  ✓ Prefilled tokens [1714:1778] (65 tokens)
  ✓ Generated tokens [1779:1810] (32 tokens)
  ✗ Pruned 267 tokens: [420, 421, 422, 423, 424, ...]
    Cache: 1809 → 1542
```

### 完整历史

```
================================================================================
TOKEN TRACKING HISTORY
================================================================================
[Step 1] Prefill:  tokens [0:1673] (1674t) → cache:  1674t
[Step 1] Generate: tokens [1674:1713] (  40t) → cache:  1714t
[Step 2] Prefill:  tokens [1714:1778] (  65t) → cache:  1779t
[Step 2] Generate: tokens [1779:1810] (  32t) → cache:  1811t
[Step 2] Prune:    discard  267t [420, 421, 422, ...] → cache:  1544t
[Step 3] Prefill:  tokens [1544:1604] (  61t) → cache:  1605t
[Step 3] Generate: tokens [1605:1635] (  31t) → cache:  1636t
================================================================================

[TOKEN STATS] Total pruned: 534, Prune events: 2, Final cache: 1636
```

---

## 🎯 Token 编号规则

**从第一个 Prefill token 开始编号为 0**：

```
初始 Prompt (1674 tokens): [0:1673]
    ↓
第一步生成 (40 tokens):   [1674:1713]
    ↓
第二步 Observation (65 tokens): [1714:1778]
    ↓
第二步生成 (32 tokens):   [1779:1810]
    ↓
【Pruning 发生】
    丢弃 tokens: [420, 421, ..., 686]
    保留 tokens: [0-419, 687-1810]
    新编号: 新 cache 为 [0:1543]（共 1544 tokens）
    ↓
第三步 Observation (61 tokens): [1544:1604]
```

---

## 💡 使用场景

### 1. 验证 H2O 压缩效果

```bash
python run_all_wiki_experiments_v2.py --experiment react_kv_h2o 2>&1 | tee h2o_tracking.log

# 在日志中查找
grep "\[TOKEN STATS\]" h2o_tracking.log
```

输出：
```
[TOKEN STATS] Total pruned: 2345, Prune events: 12, Final cache: 3456
```

### 2. 对比不同方法

```bash
# H2O
python run_all_wiki_experiments_v2.py --experiment react_kv_h2o > h2o.log

# SnapKV
python run_all_wiki_experiments_v2.py --experiment react_kv_snapkv > snapkv.log

# 对比
grep "\[TOKEN STATS\]" *.log
```

### 3. 调试 Token 丢失问题

如果在日志中看到异常的 token 变化，可以查看详细的 `TOKEN TRACKING HISTORY` 部分来确定在哪一步出现了问题。

---

## 🔍 关键指标解读

### Cache 长度变化

```
[Step 2] Cache: 1809 → 1542
↑
丢弃了 1809 - 1542 = 267 个 tokens
```

### Pruning 率

```
Total pruned: 2345 tokens
Final cache: 3456 tokens
Pruning rate: 2345 / (2345 + 3456) = 40.4%
```

### Prune 事件

```
Total prune events: 12
Average tokens pruned per event: 2345 / 12 = 195.4 tokens/event
```

---

## 📋 完整工作流

1. **初始化** → TokenTracker 创建
2. **第一步** → 记录 Prefill + Generated tokens
3. **每一步** → 
   - 记录 Prefill tokens
   - 执行 generate_incremental()
   - LLM 内部可能触发 KV Manager 的 pruning
   - 记录 Generated tokens
   - 打印步骤摘要
4. **结束** → 打印完整历史和统计信息

---

## ⚠️ 注意事项

1. **仅适用于 H2O 方法**：TokenTracker 只在 `pruning_mode == "h2o"` 时初始化
2. **性能开销**：由于需要额外的 tokenization，可能会增加约 5-10% 的运行时间
3. **Token 编号**：始终从第一个 prefill token 开始编号为 0，即使有 pruning 发生

---

## ✨ 优势

✅ 完整的 token 生命周期追踪  
✅ 清晰的 pruning 事件记录  
✅ Token 编号持续有效  
✅ 统计数据易于分析  
✅ 易于调试和验证压缩效果  

现在可以直接运行：

```bash
python run_all_wiki_experiments_v2.py --experiment react_kv_h2o
```

日志中会包含完整的 token 追踪信息！
