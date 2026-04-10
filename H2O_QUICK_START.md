# H2O 快速启动指南

## ⚡ 最快开始方式

### 1️⃣ 直接运行 H2O 实验

```bash
cd /Users/fengboyu/Documents/Python_Code/kvmem

# 单独运行 H2O
python run_all_wiki_experiments_v2.py --experiment react_kv_h2o

# 或从 Shell 脚本运行
bash run_v2_experiments.sh
```

### 2️⃣ 查看结果

```bash
# 查看完整结果
cat results/wiki_0318_v3/react_kv_h2o_wiki_500_0318.json

# 查看摘要
python run_all_wiki_experiments_v2.py --experiment collect
cat final_summary_0406_v3.md
```

---

## 🎯 核心概念

### 参数启动

所有方法都通过 `--experiment` 参数隔离：

```
python run_all_wiki_experiments_v2.py --experiment <方法>
```

| 参数 | 说明 |
|------|------|
| `single` | 单模型（无检索） |
| `rag` | RAG 检索 |
| `react` | ReAct（无 KV 缓存） |
| `react_kv_none` | ReAct + KV 缓存（无压缩） |
| **`react_kv_h2o`** | **ReAct + KV 缓存 + H2O 压缩** ✓ |
| `react_kv_snapkv` | ReAct + KV 缓存 + SnapKV |
| `ours` | ReAct + KV 缓存 + 自定义方法 |
| `all` | 运行所有方法 |
| `collect` | 生成对比报告 |

### H2O 工作流

```
Step 1: 初始化
├─ 生成 prompt KV ✓
├─ 提取 prompt_len ✓
└─ 初始化 memory_block ✓

Step 2-7: 增量生成
├─ 输入新观察和问题 ✓
├─ H2O 评分（注意力权重）✓
├─ 保留 Heavy-Hitter tokens ✓
├─ 融合到 memory_block ✓
└─ 更新 recent_kv（保护窗口）✓

最后: 输出答案
```

---

## 📊 配置参数

### H2O 配置（在 `run_react_kv_experiment()` 中）

```python
kv_config = {
    "pruning_mode": "h2o",           # ← 启用 H2O
    "keep_ratio": 0.5,               # ← 保留 50% tokens
    "prune_every_n": 2,              # ← 每 2 步压缩一次
    "observation_window": 128,       # ← 保护最近 128 tokens
    "num_score_layers": 3,           # ← 用最后 3 层评分
}
```

### 调优建议

- **提高压缩率**: 降低 `keep_ratio` (如 0.3)
- **更频繁压缩**: 降低 `prune_every_n` (如 1)
- **加大保护**: 提高 `observation_window` (如 256)

---

## 📁 关键文件

| 文件 | 作用 |
|------|------|
| `run_all_wiki_experiments_v2.py` | 主实验脚本 |
| `kv_cache/h2o_scorer.py` | H2O 评分器 |
| `kv_cache/kv_cache_manager.py` | KV 管理器 |
| `models/QwenLLMWithKVCache.py` | 带 KV 的 LLM |
| `H2O_IMPLEMENTATION.md` | 完整文档 |

---

## 🔍 监控指标

打印的 KV 结构信息：

```
[KV LEN] step=2 prompt=256 memory=[128] recent=[128]
           ↓         ↓              ↓               ↓
        当前步   初始 prompt  历史压缩状态  保护窗口
```

关键指标：
- **total_len**: KV 缓存总大小
- **memory**: 压缩后的中间状态
- **recent**: 保留的最新 tokens
- **total_prune_count**: 压缩次数

---

## ✅ 方法隔离验证

确认不同方法独立运行：

```bash
# 运行 H2O
python run_all_wiki_experiments_v2.py --experiment react_kv_h2o

# 查看日志中的配置确认
grep "pruning_mode" results/wiki_0318_v3/react_kv_h2o_wiki_500_0318.json
# 输出应该包含: "pruning_mode": "h2o"

# 与其他方法对比
python run_all_wiki_experiments_v2.py --experiment react_kv_snapkv
grep "pruning_mode" results/wiki_0318_v3/react_kv_snapkv_wiki_500_0318.json
# 输出应该包含: "pruning_mode": "snapkv"
```

---

## 🚀 完整实验流程

```bash
# 1. 运行 H2O
python run_all_wiki_experiments_v2.py --experiment react_kv_h2o

# 2. 等待完成（检查日志）
tail -f logs/logs_react_kv_h2o_wiki_0318_v2.log

# 3. 查看结果
python -c "import json; r=json.load(open('results/wiki_0318_v3/react_kv_h2o_wiki_500_0318.json')); print(f\"EM: {r['summary']['exact_match']:.1f}%, F1: {r['summary']['f1_score']:.1f}%\")"

# 4. 生成对比报告
python run_all_wiki_experiments_v2.py --experiment collect
cat final_summary_0406_v3.md
```

---

## 🐛 常见问题

**Q: 如何从中断处继续？**  
A: 脚本自动检查 checkpoint，只需重新运行相同命令

**Q: 如何修改 H2O 参数？**  
A: 编辑 `run_react_kv_experiment()` 中的 `kv_config` 字典

**Q: 结果保存在哪里？**  
A: `results/wiki_0318_v3/react_kv_h2o_wiki_500_0318.json`

**Q: 如何验证 H2O 在运行？**  
A: 检查输出日志是否包含 `[KV LEN]` 和 pruning 信息

---

## 📌 核心代码位置

**启动 H2O 的关键代码**（第 1710 行）：

```python
if args.experiment == "react_kv_h2o" or args.experiment == "all":
    run_react_kv_experiment(
        val_data, selected_samples, retriever, "h2o",  # ← "h2o" 参数
        os.path.join(output_dir, "react_kv_h2o_wiki_500_0318.json"),
        os.path.join(output_dir, "react_kv_h2o_wiki_500_0318_checkpoint.json"),
    )
```

**H2O 配置**（第 1233 行）：

```python
kv_config = {
    "pruning_mode": pruning_mode,  # ← 传入的 "h2o"
    # ... 其他配置
}
```

**方法隔离验证**：

```python
def run_react_kv_experiment(val_data, selected_samples, retriever, pruning_mode, ...):
    # pruning_mode = "h2o" 时，使用 H2O 压缩
    # pruning_mode = "snapkv" 时，使用 SnapKV 压缩
    # 其他方法不受影响
```

---

## 🎓 学习材料

- 详细设计文档: `H2O_IMPLEMENTATION.md`
- H2O 论文: Zhang et al. 2023
- ReAct 论文: Yao et al. 2022

---

**现在就可以运行**:
```bash
python run_all_wiki_experiments_v2.py --experiment react_kv_h2o
```

✨ 完成！方法已隔离，H2O 已实现
