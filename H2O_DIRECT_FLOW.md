# H2O 直接流程 - 不使用 ours 逻辑

## 📌 核心改进

**你的需求**: "H2O 的方法直接用 generate_incremental 就行，不要用 ours 的逻辑"

**已实现**: ✅ 完全分离了两种调用方式

---

## 🔄 方法流程对比

### H2O / SnapKV / None 方法
```
第一步：generate_first()
  ↓
后续步骤：generate_incremental()
  ↓
【内部自动处理】
  - 增量编码新 observation tokens
  - KVCacheManager 自动触发剪枝
  - H2O 评分器计算 Heavy-Hitter
  - 直接返回结果
  ↓
无需 memory_block 或 recent_kv 融合
直接进入下一步循环
```

### ours 方法
```
第一步：generate_first()
  ↓
初始化：memory_block, recent_kv
  ↓
后续步骤：generate_incremental_with_memory()
  ↓
【手动融合】
  - 组合 prompt_kv + memory_block + recent_kv
  - 增量编码
  - 返回分离的 obs_kv 和 gen_kv
  ↓
融合逻辑：
  - 提取 recent_kv（最后 window_size tokens）
  - 融合其余 tokens 到 memory_block
  ↓
进入下一步循环
```

---

## 🛠️ 实现细节

### 参数检测

在 `_run_react_kv_episode()` 函数中：

```python
# 根据 pruning_mode 决定是否使用 memory_block 融合
use_memory_fusion = (pruning_mode == "ours")
```

### 调用分发

```python
if use_memory_fusion:
    # ours 方法：使用融合逻辑
    response, obs_kv, gen_kv = llm.generate_incremental_with_memory(
        new_text,
        prompt_kv=prompt_kv,
        memory_block=memory_block,
        recent_kv=recent_kv,
        max_new_tokens=256,
        stop_strings=kv_stop_strings
    )
else:
    # H2O、SnapKV、None：直接调用
    response, obs_kv, gen_kv = llm.generate_incremental(
        new_text,
        max_new_tokens=256,
        stop_strings=kv_stop_strings
    )
```

### KV 融合逻辑

```python
if not use_memory_fusion:
    print(f"[INFO] Skipping KV fusion for {pruning_mode} method (auto-managed by KVCacheManager)")
    # LLM 内部已处理所有 KV 管理
else:
    # ours 方法的融合代码
    obs_kv = _normalize_kv(obs_kv, ref_kv=gen_kv)
    gen_kv = _normalize_kv(gen_kv)
    
    step_kv = []
    for (o_k, o_v), (g_k, g_v) in zip(obs_kv, gen_kv):
        k = torch.cat([o_k, g_k], dim=2)
        v = torch.cat([o_v, g_v], dim=2)
        step_kv.append((k, v))
    
    # ... 融合到 memory_block
```

### 输出监控

```python
if use_memory_fusion:
    # ours 方法：打印 memory_block 和 recent_kv
    print(f"[KV LEN] step={step} prompt={prompt_len} memory={mem_lens} recent={recent_lens}")
else:
    # H2O 等方法：直接打印总体 KV 长度
    print(f"[KV LEN] step={step} total_kv_length={kv_len}")
```

---

## 📊 H2O 内部工作流程

### 在 `QwenLLMWithKVCache` 中

**1. 第一步 - generate_first()**
```python
- 编码完整 prompt
- 生成初始 Thought/Action
- 分离 prompt_kv 和 generated_kv
- 注册到 KVCacheManager
```

**2. 后续步骤 - generate_incremental()**
```python
阶段 1: Prefill new observation tokens
  ↓
  检查是否触发剪枝：kv_manager.should_prune()
  ↓
  如果需要剪枝：
    - 调用 _do_pruning()
    - H2O 评分器评分
    - 保留 Heavy-Hitter tokens
    - 更新 past_key_values

阶段 2: Decode thought/action
  ↓
  _decode() 使用 model.generate()

阶段 3: 返回分离的 KV
  ↓
  obs_kv + gen_kv 用于日志输出（ours 方法）
  （H2O 不需要这些进行融合）
```

---

## ✅ 执行验证

### 运行 H2O 实验

```bash
python run_all_wiki_experiments_v2.py --experiment react_kv_h2o
```

**预期行为**：
```
[INFO] Step 2: Using generate_incremental (h2o method)
[INFO] Skipping KV fusion for h2o method (auto-managed by KVCacheManager)
[KV LEN] step=2 total_kv_length=512
[INFO] Step 3: Using generate_incremental (h2o method)
[INFO] Skipping KV fusion for h2o method (auto-managed by KVCacheManager)
[KV LEN] step=3 total_kv_length=450
```

### 运行 ours 实验

```bash
python run_all_wiki_experiments_v2.py --experiment ours
```

**预期行为**：
```
[INFO] Step 2: Using generate_incremental_with_memory (ours method)
[KV LEN] step=2 prompt=123 memory=[128] recent=[56]
[INFO] Step 3: Using generate_incremental_with_memory (ours method)
[KV LEN] step=3 prompt=123 memory=[128] recent=[72]
```

---

## 📝 关键优势

| 方面 | H2O (新) | ours (旧) |
|------|---------|---------|
| **调用方式** | `generate_incremental()` | `generate_incremental_with_memory()` |
| **KV 管理** | 自动 (KVCacheManager) | 手动 (memory_block + recent_kv) |
| **代码复杂度** | 低 | 高 |
| **灵活性** | 标准 | 自定义 |
| **性能** | 优 | 中 |
| **融合逻辑** | 无 | 有 |

---

## 🎯 代码改动位置

**文件**: `/Users/fengboyu/Documents/Python_Code/kvmem/run_all_wiki_experiments_v2.py`

1. **行 831-832**: 函数签名添加 `pruning_mode` 参数
2. **行 833**: 计算 `use_memory_fusion` 标志
3. **行 1337-1370**: 分发调用 `generate_incremental` vs `generate_incremental_with_memory`
4. **行 1372-1483**: 条件化 KV 融合逻辑
5. **行 1486-1512**: 条件化输出监控

---

## 💡 总结

现在 H2O 方法：
- ✅ 直接使用 `generate_incremental()`
- ✅ 无需 `memory_block` 和 `recent_kv` 融合
- ✅ 完全由 KVCacheManager 自动管理 KV 缓存
- ✅ 与 ours 方法完全隔离，互不干扰

不同方法通过 `pruning_mode` 参数自动路由到对应的流程，实现真正的参数驱动执行！
