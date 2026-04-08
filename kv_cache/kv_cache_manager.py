"""
KV Cache Manager
Top-level manager for Agent trajectory KV cache memory.
Handles cache segmentation, pruning scheduling, and position remapping.
"""

import torch
from .pruning_strategy import PruningStrategy
from .position_remapper import PositionRemapper


class KVCacheManager:
    """
    Top-level KV Cache memory manager for Agent trajectories.

    KV Cache Layout:
        [Sink Tokens] [System+Question Tokens] [Trajectory Tokens]
        <-- protected prefix -----------------> <-- prunable ---->

    The observation_window protects the last M tokens of the trajectory
    from being pruned, ensuring the model always sees recent context.
    """

    def __init__(self, config):
        """
        Args:
            config: dict with keys:
                - pruning_mode: "h2o" | "snapkv" | "h2o_snapkv" | "ours" 
                - prune_every_n: trigger pruning every N ReAct steps (default: 2)
                - max_trajectory_tokens: hard budget for trajectory region (default: 1024)
                - keep_ratio: H2O heavy hitter retention ratio (default: 0.5)
                - pool_window: SnapKV pooling window size (default: 4)
                - sink_size: Attention Sink token count (default: 4)
                - observation_window: recent tokens protected from pruning (default: 128)
                - num_score_layers: layers used for H2O scoring (default: 3)
        """
        self.pruning_mode = config.get("pruning_mode", "h2o")
        self.prune_every_n = config.get("prune_every_n", 1)
        self.max_trajectory_tokens = config.get("max_trajectory_tokens", 1024)
        self.keep_ratio = config.get("keep_ratio", 0.5)
        self.observation_window = config.get("observation_window", 128)
        self.sink_size = config.get("sink_size", 4)
        self.memory_rank = config.get("memory_rank", 128)

        self.pruning_strategy = PruningStrategy(
            mode=self.pruning_mode,
            num_score_layers=config.get("num_score_layers", 3),
            pool_window=config.get("pool_window", 4),
            memory_rank=self.memory_rank,
        )
        self.position_remapper = PositionRemapper(sink_size=self.sink_size)

        # State tracking
        self.step_count = 0
        self.total_prune_count = 0
        self.protected_prefix_len = 0  # Length of [sink + system + question]
        self.current_cache_len = 0
        self.last_pruned = False
        self.new_step_kv_lengths = 0
        self.pruning_history = []

        # Per-step segments recorded for future fusion strategies.
        # Each entry: {"id": optional id, "len": token_count}
        # The actual per-step KV blobs are provided to merge_step_kv(...) when needed.
        self.step_segments = []
        # NOTE: We do NOT keep strong references to raw KV tensors here by default
        # to avoid unexpected memory growth; callers should pass step KV into
        # merge_step_kv(...) when they want it merged into the global cache.

    def register_initial_cache(self, cache_len):
        """
        Called after the first forward pass (System Prompt + Question).
        The entire initial cache is treated as protected prefix.

        Args:
            cache_len: total length of KV cache after encoding system + question
        """
        self.protected_prefix_len = cache_len
        self.current_cache_len = cache_len
        self.step_count = 0
        self.total_prune_count = 0
        self.last_pruned = False
        self.pruning_history = []

    def append_step(self, new_tokens_len):
        """
        Called after each Thought/Action/Observation is appended to the cache.

        Args:
            new_tokens_len: number of new tokens added in this step
        """
        self.current_cache_len += new_tokens_len
        self.step_count += 1

    def merge_step_kv(self, past_key_values, step_key_values, step_token_count, step_id=None):
        """
        Merge KV chunk produced for the current ReAct step into the global past_key_values.

        Args:
            past_key_values: current full KV cache (tuple of (k,v) per layer or other supported type)
            step_key_values: KV corresponding to the current step tokens (same format as past_key_values for the step portion)
            step_token_count: number of tokens in step_key_values (sequence length)
            step_id: optional identifier for this step (for bookkeeping)

        Returns:
            new_past_key_values, new_cache_len, info
            - new_past_key_values: fused KV cache (may be the same object if fusion skipped)
            - new_cache_len: updated total cache length after merging
            - info: dict with metadata (merged: bool, reason, step_id, added_tokens, new_cache_len)

        NOTE: 当前实现为简单的按序列维度 concat（占位实现）。如果使用 DynamicCache 类型或结构不兼容，
        本方法会返回原始 past_key_values 并在 info 中说明未合并的原因。
        """
        info = {"merged": False, "step_id": step_id, "added_tokens": 0, "new_cache_len": self.current_cache_len}

        if step_token_count <= 0 or step_key_values is None:
            info["reason"] = "empty_step"
            return past_key_values, self.current_cache_len, info

        # If no existing cache, step KV becomes the cache
        if past_key_values is None:
            # Accept the step KV as the whole cache
            self.current_cache_len = step_token_count
            self.step_segments.append({"id": step_id, "len": step_token_count})
            info.update({"merged": True, "added_tokens": step_token_count, "new_cache_len": self.current_cache_len})
            return step_key_values, self.current_cache_len, info

        # Handle tuple-of-(k,v) format by concatenating along sequence dim (dim=2)
        if isinstance(past_key_values, tuple) and isinstance(step_key_values, tuple):
            try:
                new_layers = []
                for (k_old, v_old), (k_step, v_step) in zip(past_key_values, step_key_values):
                    # expected shapes:
                    # k_old: (batch, num_heads, seq_old, head_dim)
                    # k_step: (batch, num_heads, seq_step, head_dim)
                    k_new = torch.cat([k_old, k_step], dim=2)
                    v_new = torch.cat([v_old, v_step], dim=2)
                    new_layers.append((k_new, v_new))

                new_kv = tuple(new_layers)
                new_len = self.current_cache_len + step_token_count
                self.current_cache_len = new_len
                self.step_segments.append({"id": step_id, "len": step_token_count})
                info.update({"merged": True, "added_tokens": step_token_count, "new_cache_len": new_len})
                return new_kv, new_len, info
            except Exception as e:
                info["reason"] = f"concat_failed: {e}"
                return past_key_values, self.current_cache_len, info

        # DynamicCache or unknown types: do not attempt merging here (placeholder)
        info["reason"] = "unsupported_cache_type"
        return past_key_values, self.current_cache_len, info

    def get_trajectory_len(self):
        """Get current trajectory region length."""
        return self.current_cache_len - self.protected_prefix_len

    def should_prune(self):
        """
        Check if pruning should be triggered.

        Triggers when:
        1. step_count is a multiple of prune_every_n, OR
        2. trajectory region exceeds max_trajectory_tokens budget
        """
        trajectory_len = self.get_trajectory_len()

        if trajectory_len <= 0:
            return False

        # Budget exceeded: must prune
        if trajectory_len > self.max_trajectory_tokens:
            return True

        # Periodic pruning
        if self.step_count > 0 and self.step_count % self.prune_every_n == 0:
            return True

        return False

    def prune(self, past_key_values, attentions=None, step_kv=None, step_token_count=None):
        """
        执行 KV cache 的剪枝或压缩，自动识别【初始化】与【增量更新】。
        返回始终为三元组： (new_past_key_values, new_cache_len, info)
        """
        # 1. 基础状态检查
        if self.current_cache_len <= self.protected_prefix_len + self.observation_window:
            self.last_pruned = False
            return past_key_values, self.current_cache_len, {"reason": "below_protected_or_window"}

        # 2. 确定边界：Prompt 之后到最近 Window 之前
        prune_start = self.protected_prefix_len
        prune_end = self.current_cache_len - self.observation_window

        if prune_end <= prune_start:
            self.last_pruned = False
            return past_key_values, self.current_cache_len, {"reason": "no_prunable_region"}

        # 3. 自动判断初始化状态
        is_already_initialized = getattr(self, "has_initialized_memory", False)

        strategy_kwargs = {
            "past_key_values": past_key_values,
            "prune_start": prune_start,
            "prune_end": prune_end,
            "keep_ratio": self.keep_ratio,
            "new_step_kv": step_kv,
            "new_step_token_count": step_token_count,
            "total_len": self.current_cache_len,
            "window_size": self.observation_window,
            "num_layers": getattr(self.pruning_strategy, "num_score_layers", None),
            "is_initial": is_already_initialized,
        }

        # 兼容性处理：把 attentions 也放进 kwargs
        if attentions is not None:
            if isinstance(attentions, dict):
                strategy_kwargs.update(attentions)
            else:
                strategy_kwargs["attentions"] = attentions

        # 4. 尝试多种签名调用 pruning_strategy.prune，以兼容不同实现
        tried_variants = []
        results_info = {"merged": False}
        new_kv = past_key_values
        new_total_len = self.current_cache_len
        last_exc = None

        # 构造候选 kwargs 列表（按优先级）
        variants = []

        # 原始版本（带 new_step_*）
        variants.append(dict(strategy_kwargs))

        # 替换 new_step_token_count -> step_token_count, new_step_kv -> step_kv
        v2 = dict(strategy_kwargs)
        if "new_step_token_count" in v2:
            v2["step_token_count"] = v2.pop("new_step_token_count")
        if "new_step_kv" in v2:
            v2["step_kv"] = v2.pop("new_step_kv")
        variants.append(v2)

        # 移除 step 相关键的精简版本
        v3 = dict(strategy_kwargs)
        v3.pop("new_step_token_count", None)
        v3.pop("new_step_kv", None)
        variants.append(v3)

        # 最小签名：仅 past_key_values 与 attentions
        v4 = {"past_key_values": past_key_values}
        if "attentions" in strategy_kwargs:
            v4["attentions"] = strategy_kwargs["attentions"]
        elif attentions is not None:
            v4["attentions"] = attentions
        variants.append(v4)

        # 依次尝试
        for kw in variants:
            # 去重尝试
            keyset = tuple(sorted(kw.keys()))
            if keyset in tried_variants:
                continue
            tried_variants.append(keyset)
            try:
                result = self.pruning_strategy.prune(**kw)
                # 规范化返回值到 (new_kv, new_total_len, info)
                if isinstance(result, (tuple, list)):
                    if len(result) == 3:
                        new_kv, new_total_len, info = result
                    elif len(result) == 2:
                        new_kv, new_total_len = result
                        info = {"method": "prune_return_2tuple"}
                    else:
                        # 非常规返回，尽可能解析
                        new_kv = result[0]
                        new_total_len = result[1] if len(result) > 1 else self.current_cache_len
                        info = {"method": "prune_unexpected_tuple", "raw_len": len(result)}
                else:
                    # 若返回对象，尝试从属性中读取
                    info = {}
                    if hasattr(result, "past_key_values"):
                        new_kv = result.past_key_values
                    else:
                        new_kv = result
                    new_total_len = getattr(result, "new_total_len", getattr(result, "new_len", self.current_cache_len))
                    info.update({"method": "prune_return_obj"})
                # 允许 pruning_strategy 在 info 中通告初始化完成
                if isinstance(info, dict) and info.get("is_initial") is True:
                    self.has_initialized_memory = True
                results_info = info if isinstance(info, dict) else {"info": info}
                break
            except TypeError as te:
                # 参数签名不匹配，尝试下一个
                last_exc = te
                continue
            except Exception as e:
                # 记录并继续尝试其它签名
                last_exc = e
                print(f"[WARN] pruning_strategy.prune failed for keys={list(kw.keys())}: {e}")
                continue

        if new_kv is past_key_values and last_exc is not None:
            # 全部尝试失败：返回原始值并提供错误信息
            info = {"reason": "prune_all_variants_failed", "error": str(last_exc)}
            self.last_pruned = False
            return past_key_values, self.current_cache_len, info

        # 5. 更新内部维护的长度状态与历史
        try:
            self.current_cache_len = int(new_total_len)
        except Exception:
            # 保守处理
            self.current_cache_len = self.current_cache_len

        self.total_prune_count += 1
        self.last_pruned = True
        self.pruning_history.append(results_info)

        return new_kv, self.current_cache_len, results_info

    def get_stats(self):
        """Get summary statistics about cache management."""
        return {
            "total_prune_count": self.total_prune_count,
            "current_cache_len": self.current_cache_len,
            "protected_prefix_len": self.protected_prefix_len,
            "trajectory_len": self.get_trajectory_len(),
            "step_count": self.step_count,
            "pruning_history": self.pruning_history,
        }
