## H2O Token Tracking 系统解析

### 📌 重要发现

本项目实现的**不是标准 H2O 算法**，而是**增量 H2O（Incremental H2O）**版本。

---

## 🔍 关键差异

### 标准 H2O
- **一次性删除大量低分 tokens**
- 计算所有 tokens 的 attention 分数
- 根据 keep_ratio（如 0.5）一次性决定保留哪些
- 如果 keep_ratio=0.5，一次删除 ~50% 的 tokens

### 增量 H2O（本项目）
```python
# 在 h2o_scorer.py 中：
def select_heavy_hitters(self, scores, keep_ratio=0.5):
    # 每次只删最不重要的一个 token
    evict_idx = torch.argmin(scores)  # ← 只找一个最小值
    
    # 构造 keep / evict
    mask[evict_idx] = False
    heavy_hitter_indices = all_indices[mask]
    evicted_indices = all_indices[~mask]
```

**特点**：每次 pruning 只删 1 个 token，避免一次性删除大量 tokens

---

## 📊 Token Tracking 输出解读

### 例子 1：标准输出
```
[TOKEN TRACKING] Step 7:
  Current cache length: 2413
  ✓ Prefilled tokens [2349:2362] (14 tokens)
  ✓ Generated tokens [2363:2412] (50 tokens)
  [INFO] Cache length: 2413 (no pruning detected in this step)
```
**含义**：
- 这一步没有触发 pruning（`step_count % prune_every_n != 0`）
- Cache 长度正常增长：2413 = 前一步 cache + prefill + generated - pruned

### 例子 2：有 Pruning
```
[TOKEN TRACKING] Step 4:
  Current cache length: 1723
  ✓ Prefilled tokens [1903:1960] (58 tokens)
  ✓ Generated tokens [1961:1989] (29 tokens)
  ✗ Pruned 267 tokens by H2O: [58, 59, 60, 61, 62, 63, ...]
    Cache: 1990 → 1723
```
**含义**：
- Prefill + generated 后 cache = 1990
- H2O pruning 触发，删除 267 个 tokens
- 最终 cache = 1723（1990 - 267）

---

## ⏱️ Pruning 触发时机

配置：`prune_every_n=2`

```
Step 1: step_count = 1 → 1 % 2 ≠ 0 → NO PRUNE
Step 2: step_count = 2 → 2 % 2 = 0 → PRUNE ✓
Step 3: step_count = 3 → 3 % 2 ≠ 0 → NO PRUNE
Step 4: step_count = 4 → 4 % 2 = 0 → PRUNE ✓
Step 5: step_count = 5 → 5 % 2 ≠ 0 → NO PRUNE
Step 6: step_count = 6 → 6 % 2 = 0 → PRUNE ✓
```

**注意**：
- 第 1 步是初始化（prompt encoding），不算在 step_count 内
- Pruning 通常在偶数 steps 触发

---

## 🎯 Token 编号规则

**从第一个 prefill token 开始编号为 0**

```
初始 Prompt (1674 tokens): [0:1673]
↓
第 1 步生成 (40 tokens):    [1674:1713]
↓
第 2 步观察 (65 tokens):    [1714:1778]
↓
第 2 步生成 (32 tokens):    [1779:1810]
↓
[第 2 步触发 Pruning]
  删除 tokens: [某些索引范围]
  新 cache = 原 cache - 删除数
↓
第 3 步观察 (61 tokens):    [新 cache 长度 : 新 cache 长度 + 61]
```

**重要**：Pruning 后，token 编号**重新调整**（但 TokenTracker 保持全局编号）

---

## 📈 压缩统计

### 最终输出示例
```
[TOKEN STATS] Total pruned: 456, Prune events: 2, Final cache: 1734

计算：
- 总 tokens 数 = 最终 cache + 总被删除 = 1734 + 456 = 2190
- 压缩率 = 456 / 2190 = 20.8%
```

---

## ⚠️ 增量 H2O 的优缺点

### 优点 ✓
- **更稳定**：避免一次性删除大量 tokens 对模型的冲击
- **更灵活**：可以精细控制删除速度
- **更渐进**：压缩是逐步进行的

### 缺点 ✗
- **压缩速度慢**：每个 prune 事件只删 1 个 token
- **内存消耗大**：在大规模任务中可能不够有效
- **与标准 H2O 不兼容**：论文对比时需要说明

---

## 🔧 如何验证 Token Tracking 的正确性

运行这个命令查看完整历史：
```bash
python run_all_wiki_experiments_v2.py --experiment react_kv_h2o 2>&1 | grep -A 50 "TOKEN TRACKING HISTORY"
```

查看以下内容：
1. ✓ Token 编号是否从 0 开始
2. ✓ 每一步的 prefill/generated 范围是否连续
3. ✓ Pruning 步骤是否在 `step_count % 2 == 0` 时出现
4. ✓ Cache 长度是否正确反映 pruning 影响

---

## 📋 Token Tracking 的三个关键事件

### 事件 1：Prefill
```
prefill tokens [start:end] (count t) cache: total_t
```
新观察 tokens 被添加到 cache

### 事件 2：Generate  
```
generate tokens [start:end] (count t) cache: total_t
```
模型生成的新 tokens 被添加到 cache

### 事件 3：Prune
```
prune remove count t [idx1, idx2, ...] cache: after_t
```
H2O 算法删除低分 tokens，cache 长度减少

---

## 🎓 小结

- **Token Tracking 系统正确运作**
- **增量 H2O 导致每次删除 1 个 token**（这不是 bug，这是设计）
- **Pruning 频率由 `prune_every_n` 控制**
- **Token 编号全局连续**，便于追踪生命周期
