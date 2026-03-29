"""
Pruning Strategy
Unified pruning interface supporting three modes:
1. H2O-only: Score -> Evict low-scoring tokens (hard deletion)
2. SnapKV-only: Pool non-observation-window tokens (no attention scores needed)
3. H2O + SnapKV: Score -> Keep heavy hitters as-is, Pool evicted tokens

Reference: H2O (Zhang et al., 2023), SnapKV (Li et al., 2024)
"""

import torch
from transformers import DynamicCache
from .h2o_scorer import H2OScorer
from .snapkv_pooler import SnapKVPooler
from .ours import OurCompressor  # 确保已导入用户的融合器类


class PruningStrategy:
    """
    Unified pruning interface for KV cache compression.

    Modes:
        "h2o":        Use attention scores to identify heavy hitters; hard-delete the rest.
        "snapkv":     Pool all non-observation-window tokens; no attention scores needed.
        "h2o_snapkv": Use attention scores to keep heavy hitters intact;
                      pool the evicted (low-score) tokens instead of deleting them.
    """

    def __init__(self, mode="h2o_snapkv", num_score_layers=3, pool_window=4):
        """
        Args:
            mode: one of "h2o", "snapkv", "h2o_snapkv"
            num_score_layers: layers used for H2O scoring (default: 3)
            pool_window: SnapKV pooling window size (default: 4)
        """
        # 支持用户自定义的 "ours" 模式（融合/压缩占位实现）
        assert mode in ("h2o", "snapkv", "h2o_snapkv", "ours"), \
            f"Unknown pruning mode: {mode}. Must be one of: h2o, snapkv, h2o_snapkv, ours"

        self.mode = mode
        self.h2o_scorer = H2OScorer(num_score_layers=num_score_layers)
        self.snapkv_pooler = SnapKVPooler(pool_window=pool_window)

    def prune(self, past_key_values, attentions, prune_start, prune_end,
              observation_window=128, keep_ratio=0.5):
        """
        Apply the selected pruning strategy on a trajectory segment within KV cache.

        The KV cache layout is:
            [0 ... prune_start-1] = protected region (sink + question)
            [prune_start ... prune_end-1] = trajectory region (subject to pruning)
            [prune_end ... total_len-1] = observation window (recent tokens, protected)

        Args:
            past_key_values: tuple of (num_layers,) tuples, each (key, value)
                             key/value shape: (batch, num_heads, seq_len, head_dim)
            attentions: tuple of attention tensors (needed for h2o/h2o_snapkv modes).
                        Can be None for snapkv mode.
            prune_start: start index of the prunable trajectory region
            prune_end: end index of the prunable trajectory region (exclusive)
            observation_window: number of recent tokens that are never pruned
            keep_ratio: for H2O, fraction of heavy hitters to keep

        Returns:
            new_past_key_values: pruned KV cache (same format as input)
            new_total_len: new total sequence length after pruning
            pruning_info: dict with stats about the pruning operation
        """
        if prune_end <= prune_start:
            return past_key_values, self._get_cache_len(past_key_values), {"pruned": False}

        num_layers = self._get_num_layers(past_key_values)
        total_len = self._get_cache_len(past_key_values)

        # Protected prefix: [0, prune_start)
        # Prunable region: [prune_start, prune_end)
        # Protected suffix (observation window): [prune_end, total_len)
        prunable_len = prune_end - prune_start

        if prunable_len <= 0:
            return past_key_values, total_len, {"pruned": False}

        if self.mode == "h2o":
            return self._prune_h2o(
                past_key_values, attentions,
                prune_start, prune_end, total_len,
                keep_ratio, num_layers
            )
        elif self.mode == "snapkv":
            return self._prune_snapkv(
                past_key_values,
                prune_start, prune_end, total_len,
                num_layers
            )
        elif self.mode == "h2o_snapkv":
            return self._prune_h2o_snapkv(
                past_key_values, attentions,
                prune_start, prune_end, total_len,
                keep_ratio, num_layers
            )
        elif self.mode == "ours":
            return self._prune_ours(
                past_key_values, attentions,
                prune_start, prune_end, total_len,
                keep_ratio, num_layers
            )

    @staticmethod
    def _get_cache_len(past_key_values):
        """Get the sequence length from past_key_values (works with both tuple and DynamicCache)."""
        if isinstance(past_key_values, DynamicCache):
            return past_key_values.layers[0].keys.shape[2]
        else:
            k, v = past_key_values[0]
            return k.shape[2]

    @staticmethod
    def _get_kv(past_key_values, layer_idx):
        """Get (key, value) tensors for a given layer from past_key_values."""
        if isinstance(past_key_values, DynamicCache):
            return past_key_values.layers[layer_idx].keys, past_key_values.layers[layer_idx].values
        else:
            return past_key_values[layer_idx]

    @staticmethod
    def _get_device(past_key_values):
        """Get the device of the cache tensors."""
        if isinstance(past_key_values, DynamicCache):
            return past_key_values.layers[0].device
        else:
            return past_key_values[0][0].device

    @staticmethod
    def _get_num_layers(past_key_values):
        """Get the number of layers in past_key_values."""
        if isinstance(past_key_values, DynamicCache):
            return len(past_key_values.layers)
        else:
            return len(past_key_values)

    @staticmethod
    def _build_cache(new_kv_list):
        """Build a DynamicCache from a list of (key, value) tuples."""
        cache = DynamicCache()
        for layer_idx, (k, v) in enumerate(new_kv_list):
            cache.update(k, v, layer_idx)
        return cache

    def _prune_ours(self, past_key_values, attentions,
                    prune_start, prune_end, total_len,
                    keep_ratio, num_layers):
        # strategy 在此计算上下文并把必要的张量构造后传入融合器
        if prune_end <= prune_start:
            return past_key_values, total_len, {"pruned": False}

        device = self._get_device(past_key_values)
        prunable_len = prune_end - prune_start
        prunable_indices = torch.arange(prune_start, prune_end, device=device)

        # 构造 base_layers：每层 (base_k, base_v, suffix_k, suffix_v)
        base_layers = []
        for layer_idx in range(num_layers):
            k, v = self._get_kv(past_key_values, layer_idx)
            pooled_k, pooled_v = self.snapkv_pooler.pool_region(k, v, prunable_indices)

            prefix_k = k[:, :, :prune_start, :]
            prefix_v = v[:, :, :prune_start, :]
            base_k = torch.cat([prefix_k, pooled_k], dim=2)
            base_v = torch.cat([prefix_v, pooled_v], dim=2)

            suffix_k = k[:, :, prune_end:, :]
            suffix_v = v[:, :, prune_end:, :]

            base_layers.append((base_k, base_v, suffix_k, suffix_v))

        # 提取可选 step_kv 信息（由 KVCacheManager.prune/上层放入 attentions dict）
        step_kv = None
        step_token_count = 0
        if isinstance(attentions, dict):
            step_kv = attentions.get("step_kv", None)
            step_token_count = int(attentions.get("step_token_count", 0) or 0)

        compressor = OurCompressor()
        final_layers, added_tokens, used_step, note = compressor.merge(
            base_layers=base_layers,
            step_kv=step_kv,
            step_token_count=step_token_count,
        )

        # 构建 DynamicCache 并返回信息
        new_kv = self._build_cache(final_layers)
        num_pooled = self.snapkv_pooler.get_num_pooled_tokens(prunable_len)
        suffix_len = total_len - prune_end
        new_total_len = prune_start + num_pooled + (added_tokens if used_step else 0) + suffix_len

        info = {
            "pruned": True,
            "mode": "ours",
            "original_prunable": prunable_len,
            "pooled_to": int(num_pooled),
            "added_tokens": int(added_tokens),
            "new_total_len": int(new_total_len),
            "used_step_kv": bool(used_step),
            "note": note,
        }
        return new_kv, new_total_len, info

    def _prune_h2o(self, past_key_values, attentions,
                   prune_start, prune_end, total_len,
                   keep_ratio, num_layers):
        """
        H2O-only: compute scores, keep heavy hitters, hard-delete the rest.
        """
        # Compute scores for the prunable region
        scores = self.h2o_scorer.compute_scores(attentions, prune_start, prune_end)
        heavy_indices, evicted_indices = self.h2o_scorer.select_heavy_hitters(scores, keep_ratio)

        # Convert to absolute indices
        abs_heavy = heavy_indices + prune_start
        # Build final index: [prefix] + [heavy hitters] + [suffix]
        prefix_indices = torch.arange(prune_start, device=scores.device)
        suffix_indices = torch.arange(prune_end, total_len, device=scores.device)
        keep_indices = torch.cat([prefix_indices, abs_heavy, suffix_indices])

        # Reconstruct KV cache
        new_kv = []
        for layer_idx in range(num_layers):
            k, v = self._get_kv(past_key_values, layer_idx)
            new_k = k[:, :, keep_indices, :]
            new_v = v[:, :, keep_indices, :]
            new_kv.append((new_k, new_v))
        new_kv = self._build_cache(new_kv)

        new_total_len = keep_indices.shape[0]
        info = {
            "pruned": True,
            "mode": "h2o",
            "original_prunable": prune_end - prune_start,
            "kept_heavy_hitters": len(heavy_indices),
            "evicted": len(evicted_indices),
            "new_total_len": new_total_len,
        }
        return new_kv, new_total_len, info

    def _prune_snapkv(self, past_key_values,
                      prune_start, prune_end, total_len,
                      num_layers):
        """
        SnapKV-only: pool all tokens in the prunable region.
        No attention scores needed.
        """
        prunable_len = prune_end - prune_start
        prunable_indices = torch.arange(prune_start, prune_end,
                                        device=self._get_device(past_key_values))

        new_kv = []
        for layer_idx in range(num_layers):
            k, v = self._get_kv(past_key_values, layer_idx)
            # Pool the prunable region
            pooled_k, pooled_v = self.snapkv_pooler.pool_region(k, v, prunable_indices)
            # Reconstruct: [prefix] + [pooled] + [suffix]
            prefix_k = k[:, :, :prune_start, :]
            suffix_k = k[:, :, prune_end:, :]
            prefix_v = v[:, :, :prune_start, :]
            suffix_v = v[:, :, prune_end:, :]

            new_k = torch.cat([prefix_k, pooled_k, suffix_k], dim=2)
            new_v = torch.cat([prefix_v, pooled_v, suffix_v], dim=2)
            new_kv.append((new_k, new_v))
        new_kv = self._build_cache(new_kv)

        num_pooled = self.snapkv_pooler.get_num_pooled_tokens(prunable_len)
        new_total_len = prune_start + num_pooled + (total_len - prune_end)
        info = {
            "pruned": True,
            "mode": "snapkv",
            "original_prunable": prunable_len,
            "pooled_to": num_pooled,
            "new_total_len": new_total_len,
        }
        return new_kv, new_total_len, info

    def _prune_h2o_snapkv(self, past_key_values, attentions,
                          prune_start, prune_end, total_len,
                          keep_ratio, num_layers):
        """
        H2O + SnapKV: score tokens, keep heavy hitters as-is, pool the evicted ones.
        """
        # Compute scores for the prunable region
        scores = self.h2o_scorer.compute_scores(attentions, prune_start, prune_end)
        heavy_indices, evicted_indices = self.h2o_scorer.select_heavy_hitters(scores, keep_ratio)

        # Convert to absolute indices
        abs_heavy = heavy_indices + prune_start
        abs_evicted = evicted_indices + prune_start

        new_kv = []
        for layer_idx in range(num_layers):
            k, v = self._get_kv(past_key_values, layer_idx)
            # Keep heavy hitters as-is
            heavy_k = k[:, :, abs_heavy, :]
            heavy_v = v[:, :, abs_heavy, :]

            # Pool evicted tokens
            if len(abs_evicted) > 0:
                pooled_k, pooled_v = self.snapkv_pooler.pool_region(k, v, abs_evicted)
            else:
                pooled_k = k[:, :, :0, :]
                pooled_v = v[:, :, :0, :]

            # Reconstruct: [prefix] + [pooled evicted] + [heavy hitters] + [suffix]
            prefix_k = k[:, :, :prune_start, :]
            suffix_k = k[:, :, prune_end:, :]
            prefix_v = v[:, :, :prune_start, :]
            suffix_v = v[:, :, prune_end:, :]

            new_k = torch.cat([prefix_k, pooled_k, heavy_k, suffix_k], dim=2)
            new_v = torch.cat([prefix_v, pooled_v, heavy_v, suffix_v], dim=2)
            new_kv.append((new_k, new_v))
        new_kv = self._build_cache(new_kv)

        num_pooled = self.snapkv_pooler.get_num_pooled_tokens(len(evicted_indices))
        new_total_len = prune_start + num_pooled + len(heavy_indices) + (total_len - prune_end)
        info = {
            "pruned": True,
            "mode": "h2o_snapkv",
            "original_prunable": prune_end - prune_start,
            "kept_heavy_hitters": len(heavy_indices),
            "evicted": len(evicted_indices),
            "pooled_to": num_pooled,
            "new_total_len": new_total_len,
        }
        return new_kv, new_total_len, info
