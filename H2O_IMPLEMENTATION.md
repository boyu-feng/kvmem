# H2O 实现指南

## 概述

本项目已完整实现了 H2O (Heavy-Hitter Oracle) 压缩方法，并将其集成到 ReAct 框架中，使用方式如下：

```bash
PYTHON $SCRIPT --experiment react_kv_h2o
```

## 架构设计

### 方法隔离

项目采用了清晰的方法隔离架构：

```
run_all_wiki_experiments_v2.py
├── 基础方法（不使用 KV Cache）
│   ├── run_single_experiment()      # 单模型
│   ├── run_rag_experiment()         # RAG
│   └── run_react_experiment()       # ReAct
│
└── KV Cache 方法（统一函数，通过 pruning_mode 参数隔离）
    └── run_react_kv_experiment(pruning_mode)
        ├── "none"       → ReAct-KV (无压缩)
        ├── "h2o"        → ReAct-KV (H2O 压缩)  ✓ 主要实现
        ├── "snapkv"     → ReAct-KV (SnapKV 压缩)
        └── "ours"       → ReAct-KV (自定义方法)
```

### 关键数据流

#### 第一步（初始化）
```
Question + Prompt
    ↓
generate_first()  ← llm.generate_first()
    ↓
(response, prompt_kv, generated_kv)
    ↓
_process_kv_flow() 处理初始 KV
    ↓
(recent_kv, memory_block) ← 初始化完成
```

#### 后续步骤（增量生成）
```
Observation + Thought
    ↓
generate_incremental_with_memory(
    new_text,
    prompt_kv,           # 初始 prompt 的 KV
    memory_block,        # 历史中间状态
    recent_kv            # 最近 tokens（保留窗口）
)
    ↓
(response, obs_kv, gen_kv)
    ↓
KV 融合与 H2O 压缩
    ↓
(recent_kv, memory_block) ← 更新
```

## H2O 配置

KV Cache 的 H2O 配置在 `run_react_kv_experiment()` 中定义：

```python
kv_config = {
    "pruning_mode": "h2o",           # 启用 H2O 压缩
    "prune_every_n": 2,              # 每 2 步触发一次压缩
    "keep_ratio": 0.5,               # 保留 50% 的 heavy-hitter tokens
    "pool_window": 4,                # SnapKV 池化窗口（不影响 H2O）
    "max_trajectory_tokens": 1024,   # 轨迹 KV 的最大长度
    "sink_size": 4,                  # 注意力 Sink tokens 数量
    "observation_window": 128,       # 保护窗口大小（最近 128 tokens）
    "num_score_layers": 3,           # H2O 评分使用最后 3 层
    "attn_mode": "scoring_forward",  # H2O 评分模式
}
```

## H2O 核心模块

### 1. H2O 评分器 (`h2o_scorer.py`)

```python
class H2OScorer:
    def __init__(self, num_score_layers=3):
        # 使用最后 N 层的注意力权重进行评分
        pass
    
    def compute_scores(self, attentions, start_pos, end_pos):
        # 计算累积注意力分数
        # 输出：(end_pos - start_pos,) 的重要性分数
        pass
    
    def select_heavy_hitters(self, scores, keep_ratio=0.5):
        # 选择得分最高的 tokens
        # 增量删除最不重要的 token（改进方案）
        pass
```

### 2. KV Cache 管理器 (`kv_cache_manager.py`)

```python
class KVCacheManager:
    def __init__(self, config):
        self.pruning_strategy = PruningStrategy(mode="h2o", ...)
        self.position_remapper = PositionRemapper(...)
        # 初始化压缩和位置映射策略
    
    def register_initial_cache(self, cache_len):
        # 注册初始缓存（受保护）
        pass
    
    def prune_by_score(self, h2o_scores):
        # 基于 H2O 分数进行压缩
        pass
```

### 3. 位置重映射 (`position_remapper.py`)

在压缩后重新映射 position embeddings：

```
原始位置: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
H2O 删除: [2, 5, 8] (不重要的 tokens)
新位置:  [0, 1, 2, 3, 4, 5, 6]
重映射:  [0→0, 1→1, 3→2, 4→3, 6→4, 7→5, 9→6]
```

## 运行命令

### 单独运行 H2O 实验

```bash
# 基本运行
CUDA_VISIBLE_DEVICES=0 python run_all_wiki_experiments_v2.py --experiment react_kv_h2o

# 使用自定义输出目录
CUDA_VISIBLE_DEVICES=0 python run_all_wiki_experiments_v2.py --experiment react_kv_h2o --output_dir results/h2o_test

# 从检查点恢复（自动）
# 脚本会检查 checkpoint 文件，继续未完成的样本
```

### 从 Shell 脚本运行

```bash
# 编辑 run_v2_experiments.sh，确保以下行未被注释：
# echo "$(date): Starting v2 ReAct-KV (H2O) experiment..."
# $PYTHON $SCRIPT --experiment react_kv_h2o > ${LOGDIR}/logs_react_kv_h2o_wiki_0318_v2.log 2>&1
# echo "$(date): v2 ReAct-KV (H2O) done."

bash run_v2_experiments.sh
```

## 输出结果

### 结果文件位置

```
results/wiki_0318_v3/
├── react_kv_h2o_wiki_500_0318.json           # 完整结果
└── react_kv_h2o_wiki_500_0318_checkpoint.json # 检查点
```

### 结果结构

```json
{
  "summary": {
    "method": "ReAct-KV (h2o, Full Wiki)",
    "exact_match": 45.23,
    "f1_score": 58.47,
    "total_time_seconds": 3600,
    "pruning_mode": "h2o",
    "kv_config": {...},
    "timing_stats": {
      "avg_kv_cache_length": 156,
      "max_kv_cache_length": 512,
      "total_prune_count": 250
    }
  },
  "results": [
    {
      "id": "...",
      "question": "...",
      "predicted_answer": "...",
      "em": 1,
      "f1": 0.95,
      "num_steps": 3,
      "llm_stats": {
        "pruning_time": 0.023,
        "total_prune_count": 12
      }
    }
  ]
}
```

## 方法对比

| 方法 | 命令 | 特点 |
|------|------|------|
| 单模型 | `--experiment single` | 直接 QA，无检索 |
| RAG | `--experiment rag` | BM25 检索 + 阅读 |
| ReAct | `--experiment react` | 多步推理，无 KV 缓存 |
| ReAct-KV (无压缩) | `--experiment react_kv_none` | KV 缓存复用，无压缩 |
| **ReAct-KV (H2O)** | **`--experiment react_kv_h2o`** | **KV 缓存 + H2O 压缩** ✓ |
| ReAct-KV (SnapKV) | `--experiment react_kv_snapkv` | KV 缓存 + SnapKV 压缩 |
| ReAct-KV (Ours) | `--experiment ours` | KV 缓存 + 自定义压缩 |

## 监控和调试

### 日志输出

在生成过程中，会打印详细的 KV 信息：

```
[KV LEN] step=1 prompt=256 memory=[] recent=[256]
[KV STRUCT] step=1
  total_len=256
  prompt: [0:256] len=256
  memory: [256:256] len=0
  recent: [256:256] len=256

[KV LEN] step=2 prompt=256 memory=[128] recent=[128]
[KV STRUCT] step=2
  total_len=512
  prompt: [0:256] len=256
  memory: [256:384] len=128
  recent: [384:512] len=128
```

### 关键指标

- **total_len**: KV 缓存总长度
- **memory**: 历史压缩状态的长度
- **recent**: 保留窗口（最近 tokens）的长度
- **pruning_count**: 压缩次数

## 常见问题

### Q1: 如何调整 H2O 的压缩强度？

修改 `run_react_kv_experiment()` 中的配置：

```python
kv_config = {
    "keep_ratio": 0.5,      # 降低此值以提高压缩率
    "prune_every_n": 1,     # 降低此值以更频繁地压缩
    "observation_window": 64,  # 降低此值以减少保护窗口
}
```

### Q2: H2O 与 SnapKV 的主要区别是什么？

- **H2O**: 基于**注意力权重**的重要性选择（Heavy-Hitter Oracle）
- **SnapKV**: 基于**局部池化**的结构化压缩（Attention Pooling）
- **Ours**: 可能是结合两者或采用其他策略

### Q3: 如何对比所有方法？

```bash
# 运行所有实验
PYTHON run_all_wiki_experiments_v2.py --experiment all

# 生成对比报告
PYTHON run_all_wiki_experiments_v2.py --experiment collect
```

## 相关论文参考

- **H2O**: Zhang et al., 2023 - "H2O: Heavy-Hitter Oracle for Efficient Generative Inference"
- **ReAct**: Yao et al., 2022 - "ReAct: Synergizing Reasoning and Acting in Language Models"
- **SnapKV**: Li et al., 2024 - "SnapKV: LLM Knows What It Needs"

## 项目结构

```
kvmem/
├── kv_cache/
│   ├── h2o_scorer.py           # H2O 评分器 ✓
│   ├── kv_cache_manager.py     # KV 管理器 ✓
│   ├── position_remapper.py    # 位置重映射 ✓
│   ├── pruning_strategy.py     # 压缩策略 ✓
│   └── ours.py                 # 自定义融合器 ✓
├── models/
│   ├── QwenLLM.py              # 基础模型
│   └── QwenLLMWithKVCache.py   # KV Cache 模型 ✓
├── retrievers/
│   ├── BM25Retriever.py
│   └── WikiBM25Retriever.py
├── run_all_wiki_experiments_v2.py  # 主实验脚本 ✓
├── run_v2_experiments.sh           # Shell 运行脚本 ✓
└── H2O_IMPLEMENTATION.md           # 本文件
```

## 下一步

1. **运行 H2O 实验**: `PYTHON run_all_wiki_experiments_v2.py --experiment react_kv_h2o`
2. **对比结果**: 查看 `react_kv_h2o_wiki_500_0318.json`
3. **微调配置**: 调整 `keep_ratio` 和 `prune_every_n` 参数以找到最佳效果
4. **生成报告**: `PYTHON run_all_wiki_experiments_v2.py --experiment collect`

---

**最后更新**: 2026-04-10
**实现版本**: 完整隔离设计，支持 H2O、SnapKV 和自定义方法

