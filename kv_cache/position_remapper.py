"""
Position Remapper
Handles position ID management after KV cache pruning.
Reference: StreamingLLM - Xiao et al., 2023

Current implementation: simple sequential remapping without RoPE re-rotation.
The model receives contiguous position IDs for the pruned cache.
"""

import torch


class PositionRemapper:
    """
    Manages position IDs for KV cache after pruning.

    Simple version (no RoPE re-rotation):
    After pruning, assign sequential position IDs to all remaining tokens.
    This avoids the complexity of RoPE inverse-rotation on cached keys.
    """

    def __init__(self, sink_size=4):
        """
        Args:
            sink_size: Number of initial tokens treated as Attention Sinks.
                       These tokens are always preserved during pruning.
        """
        self.sink_size = sink_size

    def remap(self, total_kept_tokens):
        """
        Generate sequential position IDs for kept tokens.

        Args:
            total_kept_tokens: total number of tokens remaining after pruning
                               (including sink, question, pruned trajectory, etc.)

        Returns:
            new_position_ids: Tensor of shape (1, total_kept_tokens),
                              sequential IDs [0, 1, 2, ..., total_kept_tokens-1]
            next_position: the position ID for the next new token to be appended
        """
        new_position_ids = torch.arange(total_kept_tokens).unsqueeze(0)  # (1, N)
        next_position = total_kept_tokens
        return new_position_ids, next_position

    def get_next_positions(self, current_cache_len, new_token_count):
        """
        Get position IDs for newly appended tokens.

        Args:
            current_cache_len: current length of the KV cache
            new_token_count: number of new tokens to append

        Returns:
            position_ids: Tensor of shape (1, new_token_count)
        """
        positions = torch.arange(
            current_cache_len,
            current_cache_len + new_token_count
        ).unsqueeze(0)  # (1, new_token_count)
        return positions
