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

    def __init__(self, config, token_tracker=None):
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
            token_tracker: optional TokenTracker instance for tracking pruned tokens
        """
        self.pruning_mode = config.get("pruning_mode", "h2o")
        self.prune_every_n = config.get("prune_every_n", 1)
        self.max_trajectory_tokens = config.get("max_trajectory_tokens", 1024)
        self.keep_ratio = config.get("keep_ratio", 0.5)
        self.observation_window = config.get("observation_window", 128)
        self.sink_size = config.get("sink_size", 4)
        self.memory_rank = config.get("memory_rank", 128)
        self.token_tracker = token_tracker

        self.pruning_strategy = PruningStrategy(
            mode=self.pruning_mode,
            num_score_layers=config.get("num_score_layers", 3),
            pool_window=config.get("pool_window", 4),
            memory_rank=self.memory_rank,
            token_tracker=token_tracker,
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
        
        # Initialize token tracker if available
        if self.token_tracker is not None:
            self.token_tracker.set_initial_cache_length(cache_len)

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

    def prune(self, past_key_values, attentions=None):
        """
        Execute pruning on the trajectory region of the KV cache.

        Args:
            past_key_values: full KV cache tuple
            attentions: attention weights (needed for h2o/h2o_snapkv modes)

        Returns:
            new_past_key_values: pruned KV cache
            new_cache_len: new total cache length
        """
        trajectory_len = self.get_trajectory_len()
        if trajectory_len <= 0:
            self.last_pruned = False
            return past_key_values, self.current_cache_len

        # Determine prunable region boundaries
        prune_start = self.protected_prefix_len

        # Protect the observation window (recent tokens)
        obs_window = min(self.observation_window, trajectory_len)
        prune_end = self.current_cache_len - obs_window

        if prune_end <= prune_start:
            # Not enough tokens to prune (all within observation window)
            self.last_pruned = False
            return past_key_values, self.current_cache_len

        # Execute pruning
        cache_before = self.current_cache_len  # Track cache size before pruning
        new_kv, new_total_len, info = self.pruning_strategy.prune(
            past_key_values=past_key_values,
            attentions=attentions,
            prune_start=prune_start,
            prune_end=prune_end,
            observation_window=obs_window,
            keep_ratio=self.keep_ratio,
        )

        # Add cache_before to info for tracking
        info["cache_before"] = cache_before
        
        # Update state
        self.current_cache_len = new_total_len
        self.total_prune_count += 1
        self.last_pruned = True
        self.pruning_history.append(info)

        return new_kv, new_total_len

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
