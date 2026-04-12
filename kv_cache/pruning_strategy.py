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
from .ours import OurCompressor  # 确保已导入用户的融合器类


class PruningStrategy:
    """
    Unified pruning interface for KV cache compression.

    Modes:
        "h2o":        Use attention scores to identify heavy hitters; hard-delete the rest.
        "snapkv":     Pool all non-observation-window tokens; no attention scores needed.
        "h2o_snapkv": Use attention scores to keep heavy hitters intact;
                      pool the evicted (low-score) tokens instead of deleting them.
        "ours":       Our proposed method that integrates scoring, pooling, and optional step KV fusion.
    """

    def __init__(self, mode="ours", num_score_layers=3, pool_window=4, memory_rank=128, token_tracker=None):
        """
        Args:
            mode: one of "h2o", "snapkv", "h2o_snapkv", "ours"
            num_score_layers: layers used for H2O scoring (default: 3)
            pool_window: SnapKV pooling window size (default: 4)
            memory_rank: fixed rank for Memory base (used in "ours" mode)
            token_tracker: optional TokenTracker instance for tracking pruned tokens
        """
        # 支持用户自定义的 "ours" 模式（融合/压缩占位实现）
        assert mode in ("h2o", "snapkv", "h2o_snapkv", "ours"), \
            f"Unknown pruning mode: {mode}. Must be one of: h2o, snapkv, h2o_snapkv, ours"

        self.mode = mode
        self.h2o_scorer = H2OScorer(num_score_layers=num_score_layers)
        self.memory_rank =  memory_rank # 仅在 "ours" 模式下使用，表示 Memory 基底的固定秩
        self.token_tracker = token_tracker

    def prune(self, past_key_values, attentions, prune_start, prune_end, new_step_kv=None,step_token_count=0, keep_ratio=0.5,
              observation_window=128,is_initial=False):
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
            print(f"[INFO] Running H2O pruning: scoring and hard-deleting low-scoring tokens")
            return self._prune_h2o(
                past_key_values, attentions,
                prune_start, prune_end, total_len,
                keep_ratio, num_layers
            )
        elif self.mode == "snapkv":
            print(f"[INFO] Running SnapKV pruning: pooling and selecting top-k tokens")
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
                prune_start, prune_end, new_step_kv, total_len,
                keep_ratio, num_layers,window_size=observation_window,is_initial=is_initial
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
                    prune_start, prune_end, new_step_kv=None, total_len=None,
                    new_step_token_count=0,
                    keep_ratio=None, window_size=512, num_layers=None,
                    is_initial=False # 初始传入为 False
                    ):
        
        if window_size is None:
            window_size = 512

        # memory_rank 是你预设的固定记忆矩阵长度（如 128）
        memory_rank = self.memory_rank
        device = self._get_device(past_key_values)
        
        layers_data = []
        
        for layer_idx in range(num_layers):
            k, v = self._get_kv(past_key_values, layer_idx)
            
            # --- 【变量 A：静态前缀 (Prompt)】 ---
            # 绝对不动，保持模型对指令的遵循能力
            prefix_k = k[:, :, :prune_start, :]
            prefix_v = v[:, :, :prune_start, :]

            # --- 【变量 B：长期记忆块 (Memory Block M)】 ---
            if not is_initial:
                # 情况 1：第一次压缩初始化
                # 此时：[Prompt] + [第一次输出的内容] + [新 Step] + [Window]
                # 我们要把“第一次输出的内容”池化成 M
                # 范围计算：从 Prompt 结束到 (窗口前 - 新 Step 前)
                hist_end = total_len - window_size - new_step_token_count
                
                if hist_end > prune_start:
                    prunable_indices = torch.arange(prune_start, hist_end, device=device)
                    # 利用池化把第一次生成的 Thought/Action 压成固定秩 r
                    memory_m_k, memory_m_v = self.snapkv_pooler.pool_region(k, v, prunable_indices)
                else:
                    # 容错：如果第一次输出太短，直接截取原始段作为初始 M
                    memory_m_k = k[:, :, prune_start:hist_end, :]
                    memory_m_v = v[:, :, prune_start:hist_end, :]
                
                layer_initialized_status = False # 标记此层刚完成初始化，待融合
            else:
                # 情况 2：增量更新模式
                # 此时 M 已经在固定位置：[prune_start : prune_start + memory_rank]
                m_start, m_end = prune_start, prune_start + memory_rank
                memory_m_k = k[:, :, m_start:m_end, :]
                memory_m_v = v[:, :, m_start:m_end, :]
                
                layer_initialized_status = True

            # --- 【变量 C：新步进信号 (Step Signal S)】 ---
            # 这是本次 Incremental Step 产生的 Observation
            if new_step_kv is not None:
                new_s_k, new_s_v = new_step_kv[layer_idx]
            else:
                # 如果外部没传，从 Window 之前精准切出这一步的长度
                s_end = total_len - window_size
                s_start = max(0, s_end - new_step_token_count)
                new_s_k = k[:, :, s_start:s_end, :].detach()
                new_s_v = v[:, :, s_start:s_end, :].detach()

            # --- 【变量 D：原始滑动窗口 (Window W)】 ---
            # 保持 100% 精度，不做任何处理
            window_k = k[:, :, -window_size:, :]
            window_v = v[:, :, -window_size:, :]
            
            # 打包本层数据，准备喂给融合函数
            layers_data.append({
                "prefix": (prefix_k, prefix_v),
                "memory_m": (memory_m_k, memory_m_v), # 长期记忆 M (旧)
                "new_step_s": (new_s_k, new_s_v),     # 本次增量 S
                "window": (window_k, window_v),       # 滑动窗口 W
                "is_initialized": layer_initialized_status
            })

        # --- 【核心：融合与更新】 ---
        # 调用融合函数：M_new = f(M_old, S)
        # 你可以在这个函数里写 Delta Rule 或者简单的 Concat
        updated_layers = self._fuse_memory_and_signal(layers_data)

        # --- 【重新拼接 (Reconstruct)】 ---
        reconstructed_layers = []
        for data in updated_layers:
            pk, pv = data["prefix"]
            mk, mv = data["memory_m"] # 融合后的新记忆 M_new
            wk, wv = data["window"]
            
            # 布局：[Prompt] + [Memory Block] + [Sliding Window]
            merged_k = torch.cat([pk, mk, wk], dim=2)
            merged_v = torch.cat([pv, mv, wv], dim=2)
            reconstructed_layers.append((merged_k, merged_v))

        # 构建新的 Cache 对象
        new_kv = self._build_cache(reconstructed_layers)
        
        # 逻辑总长度：前缀长 + 记忆秩 + 窗口长
        new_total_len = prune_start + memory_rank + window_size

        # 告知外部：初始化已完成，下次请设为 True
        return new_kv, new_total_len, {"is_initial": True, "note": "Memory Initialized and Fused"}


    def _fuse_memory_and_signal(self, layers_data):
        """
        简单的记忆融合函数（用于跑通架构）。
        逻辑：如果已初始化，则将新信号 S 池化到 memory_rank 长度后与 M 相加。
        """
        updated_layers = []

        for layer_idx, data in enumerate(layers_data):
            # 提取变量
            (m_k, m_v) = data["memory_m"]      # 长期记忆 [B, H, rank, D]
            (s_k, s_v) = data["new_step_s"]    # 新步进信号 [B, H, step_len, D]
            is_init = data["is_initialized"]   # 是否已经建立过记忆

            # 如果是第一次初始化 (is_init 为 False)，memory_m 已经是池化好的初始基底
            # 我们直接使用它，不做额外融合
            if not is_init:
                new_m_k, new_m_v = m_k, m_v
            else:
                # 如果是增量更新 (is_init 为 True)
                # 为了跑通逻辑，我们将新信号 S 也池化成 memory_rank 的长度
                if s_k is not None and s_k.size(2) > 0:
                    # 简单的池化对齐：将 S 压缩到与 M 相同的 rank
                    # 这里的 pool_tensor 是你 SnapKV 里的基础操作
                    s_k_pooled = self.snapkv_pooler.pool_tensor(s_k, self.memory_rank)
                    s_v_pooled = self.snapkv_pooler.pool_tensor(s_v, self.memory_rank)
                    
                    # 执行最简单的融合：M_new = M_old + S_pooled
                    new_m_k = m_k + s_k_pooled
                    new_m_v = m_v + s_v_pooled
                else:
                    # 如果没有新信号，保持原样
                    new_m_k, new_m_v = m_k, m_v

            # 更新本层数据
            data["memory_m"] = (new_m_k, new_m_v)
            updated_layers.append(data)

        return updated_layers

    def _prune_h2o(self, past_key_values, attentions,
                   prune_start, prune_end, total_len,
                   keep_ratio, num_layers):
        """
        H2O-only: compute scores over entire prunable region, keep heavy hitters, hard-delete the rest.
        
        KEY CHANGE: H2O should score the ENTIRE prunable region [prune_start, prune_end),
        not just the most recent tokens. This is the core H2O algorithm: evaluate all
        tokens and keep those with highest attention scores globally.
        """
        # ================================================================
        # H2O scores the entire prunable region for importance evaluation
        # ================================================================
        print(f"[DEBUG] H2O: Computing importance scores for entire region [{prune_start}, {prune_end})")
        print(f"[DEBUG] H2O: Prunable region size: {prune_end - prune_start} tokens")
        
        scores = self.h2o_scorer.compute_scores(attentions, prune_start, prune_end)
        heavy_indices, evicted_indices = self.h2o_scorer.select_heavy_hitters(scores, keep_ratio)
        
        print(f"[DEBUG] H2O: Evaluated {scores.shape[0]} tokens in prunable region")
        print(f"[DEBUG] H2O: Keep ratio {keep_ratio:.2f} -> Keep {len(heavy_indices)} / Evict {len(evicted_indices)}")
        
        # Convert relative indices (within prunable region) to absolute indices (in full cache)
        abs_heavy = heavy_indices + prune_start
        
        # Build final index: [prefix] + [selected heavy hitters] + [suffix]
        # prefix: [0, prune_start) - protected prefix (system prompt, question)
        # selected: chosen heavy hitter tokens from [prune_start, prune_end)
        # suffix: [prune_end, total_len) - observation window (recent tokens, protected)
        prefix_indices = torch.arange(prune_start, device=scores.device)
        suffix_indices = torch.arange(prune_end, total_len, device=scores.device)
        keep_indices = torch.cat([prefix_indices, abs_heavy, suffix_indices])

        # Reconstruct KV cache with selected tokens
        new_kv = []
        for layer_idx in range(num_layers):
            k, v = self._get_kv(past_key_values, layer_idx)
            new_k = k[:, :, keep_indices, :]
            new_v = v[:, :, keep_indices, :]
            new_kv.append((new_k, new_v))
        new_kv = self._build_cache(new_kv)

        new_total_len = keep_indices.shape[0]
        
        # Convert keep_indices to a CPU list for token tracker
        kept_indices_list = keep_indices.cpu().tolist() if hasattr(keep_indices, 'cpu') else list(keep_indices)
        
        # Record to token tracker if available
        if self.token_tracker is not None:
            self.token_tracker.record_pruning_with_kept_indices(
                step=None,  # Step number will be set by caller
                kept_local_indices=kept_indices_list,
                old_cache_length=total_len
            )
        
        info = {
            "pruned": True,
            "mode": "h2o",
            "prunable_region_size": prune_end - prune_start,
            "heavy_hitters_kept": len(heavy_indices),
            "tokens_evicted": len(evicted_indices),
            "compression_ratio": len(heavy_indices) / max(1, prune_end - prune_start),
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
