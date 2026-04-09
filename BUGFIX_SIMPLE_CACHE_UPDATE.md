# 修复 _SimpleDynamicCache.update 方法缺失问题

## 问题描述

在运行第 3 步及以后的 ReAct 循环时，发生以下错误：

```
AttributeError: '_SimpleDynamicCache' object has no attribute 'update'
```

错误出现在文件 `models/QwenLLMWithKVCache.py` 的 `generate_incremental_with_memory` 函数中，当 Qwen2 模型的前向传递尝试调用 `past_key_values.update()` 方法时发生。

## 根本原因

在 Qwen2 模型的前向传递过程中，模型代码会调用缓存对象的 `update` 方法来追加新的 key/value 状态。我们创建的 fallback 类 `_SimpleDynamicCache` 虽然实现了其他必要的接口方法（如 `get_seq_length()`, `crop()`, `get_mask_sizes()`, `device` 属性），但缺少了 `update` 方法。

## 解决方案

在 `models/QwenLLMWithKVCache.py` 的 `_SimpleDynamicCache` 类中添加了 `update` 方法：

```python
def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
    """
    Update the cache with new key and value states for a specific layer.
    This method is called by the model during forward pass.
    
    Args:
        key_states: new key tensor to append
        value_states: new value tensor to append
        layer_idx: index of the layer being updated
        cache_kwargs: optional kwargs (unused in this simple implementation)
    
    Returns:
        Tuple of (updated_key_states, updated_value_states)
    """
    # 确保层索引有效
    while len(self.layers) <= layer_idx:
        self.layers.append(_SimpleLayer(
            torch.empty((1, 1, 0, key_states.shape[-1]), 
                      dtype=key_states.dtype, device=key_states.device),
            torch.empty((1, 1, 0, value_states.shape[-1]), 
                      dtype=value_states.dtype, device=value_states.device)
        ))
    
    # 将新的 key/value 与现有缓存连接
    if self.layers[layer_idx].keys.shape[2] == 0:
        # 空缓存，初始化新状态
        self.layers[layer_idx].keys = key_states.detach().clone()
        self.layers[layer_idx].values = value_states.detach().clone()
    else:
        # 追加到现有缓存
        self.layers[layer_idx].keys = torch.cat(
            [self.layers[layer_idx].keys, key_states], dim=2
        ).detach().clone()
        self.layers[layer_idx].values = torch.cat(
            [self.layers[layer_idx].values, value_states], dim=2
        ).detach().clone()
    
    # 返回完整的缓存 key/value
    return self.layers[layer_idx].keys, self.layers[layer_idx].values
```

### 方法的关键特性

1. **层索引自动扩展**：如果 `layer_idx` 超出当前层数，会自动创建空层
2. **正确的张量连接**：沿着序列维度（dim=2）连接新的张量，保持批处理和头维度不变
3. **设备和数据类型兼容性**：创建的空层使用与新张量相同的设备和数据类型
4. **返回值格式**：返回完整的缓存键值对，这是 Qwen2 模型前向传递期望的格式

## 验证

创建了单元测试 `test_simple_cache.py` 来验证 `update` 方法的功能，测试结果表明：

- ✓ 缓存初始化正确
- ✓ update 方法能够正确追加新的 key/value
- ✓ 层自动扩展功能正常
- ✓ get_mask_sizes 和 device 属性正常工作
- ✓ 所有张量操作都在正确的设备上进行

## 预期效果

修复后，当使用 fallback `_SimpleDynamicCache` 时（即当 transformers 库中 `DynamicCache.from_legacy_cache` 不可用时），模型应该能够正常进行前向传递，不再发生 `AttributeError`。这使得 ReAct 循环能够继续进行，允许第 3 步及以后的推理步骤正常执行。

## 修改的文件

- `/Users/fengboyu/Documents/Python_Code/kvmem/models/QwenLLMWithKVCache.py`
  - 在 `_SimpleDynamicCache` 类中添加了 `update` 方法

- `/Users/fengboyu/Documents/Python_Code/kvmem/test_simple_cache.py`（新增）
  - 单元测试文件，验证 `_SimpleDynamicCache.update` 方法的正确性
