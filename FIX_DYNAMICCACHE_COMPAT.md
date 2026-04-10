# 修复：DynamicCache 兼容性问题

## 🐛 问题

在 `generate_incremental()` 方法中出现错误：

```
ValueError: too many values to unpack (expected 2)
  File "QwenLLMWithKVCache.py", line 604, in generate_incremental
    k, v = layer_pkv
```

## 原因

`self.past_key_values` 可能是两种格式：

1. **Tuple 格式**（传统）：`((k, v), (k, v), ...)`
   - 直接迭代得到 (k, v) 元组

2. **DynamicCache 格式**（新版 transformers）：
   - 有 `.layers` 属性，每个 layer 是对象
   - 直接迭代得到 layer 对象，不是 (k, v) 元组
   - 需要访问 `layer.keys` 和 `layer.values`

## ✅ 解决方案

在提取 KV 的两个地方添加类型检测：

### 位置 1: 提取 obs_kv（第 604-615 行）

```python
# 提取这一轮 Observation 的 KV (step_kv)
obs_kv = []
if hasattr(outputs.past_key_values, "layers"):
    # DynamicCache 格式
    for layer in outputs.past_key_values.layers:
        k = layer.keys
        v = layer.values
        s_k = k[:, :, -new_token_count:, :].detach().clone()
        s_v = v[:, :, -new_token_count:, :].detach().clone()
        obs_kv.append((s_k, s_v))
else:
    # Tuple 格式
    for layer_pkv in outputs.past_key_values:
        k, v = layer_pkv
        s_k = k[:, :, -new_token_count:, :].detach().clone()
        s_v = v[:, :, -new_token_count:, :].detach().clone()
        obs_kv.append((s_k, s_v))
obs_kv = tuple(obs_kv)
```

### 位置 2: 提取 gen_kv（第 629-648 行）

```python
# 提取模型生成的 Thought/Action 的 KV (gen_kv)
gen_kv = []
if generated_len > 0:
    if hasattr(self.past_key_values, "layers"):
        # DynamicCache 格式
        for layer in self.past_key_values.layers:
            k = layer.keys
            v = layer.values
            g_k = k[:, :, -generated_len:, :].detach().clone()
            g_v = v[:, :, -generated_len:, :].detach().clone()
            gen_kv.append((g_k, g_v))
    else:
        # Tuple 格式
        for layer_pkv in self.past_key_values:
            k, v = layer_pkv
            g_k = k[:, :, -generated_len:, :].detach().clone()
            g_v = v[:, :, -generated_len:, :].detach().clone()
            gen_kv.append((g_k, g_v))
gen_kv = tuple(gen_kv)
```

## 📝 修改文件

**文件**: `/Users/fengboyu/Documents/Python_Code/kvmem/models/QwenLLMWithKVCache.py`

- 第 602-616 行: 修复 obs_kv 提取逻辑
- 第 626-648 行: 修复 gen_kv 提取逻辑

## 🎯 影响范围

- ✅ 修复 H2O、SnapKV、None 方法的 `generate_incremental()` 调用
- ✅ 兼容 DynamicCache 和 Tuple 两种 KV 格式
- ✅ 不影响 ours 方法的 `generate_incremental_with_memory()`

## ✨ 下一步

现在可以再次运行：

```bash
python run_all_wiki_experiments_v2.py --experiment react_kv_h2o
```

应该不会再出现 `ValueError: too many values to unpack` 的错误。
