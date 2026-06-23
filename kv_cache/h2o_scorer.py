"""
H2O (Heavy-Hitter Oracle) Scorer
Computes cumulative attention scores for each KV cache position.
Reference: H2O - Zhang et al., 2023
"""

import torch


class H2OScorer:
    """
    Computes cumulative attention scores for each KV position.
    Uses the last N layers' attention weights to determine token importance.
    Heavy-hitter tokens (high cumulative attention) are kept during pruning.
    """

    def __init__(self, num_score_layers=3):
        """
        Args:
            num_score_layers: Number of last layers to use for scoring.
                              Default=3, following SnapKV's observation that
                              last few layers capture the most relevant patterns.
        """
        self.num_score_layers = num_score_layers

    def compute_scores(self, attentions, start_pos=None, end_pos=None):
        """
        Compute importance scores for KV positions.

        Args:
            attentions: tuple of (num_layers,) tensors, each shape
                        (batch, num_heads, query_len, kv_len)
                        - From model forward with output_attentions=True
                        - NOTE: If from scoring forward (dummy token), kv_len may be
                          cache_len+1 (includes the dummy). The caller should set
                          end_pos to exclude the dummy token position.
            start_pos: start of region to score (inclusive). None = 0
            end_pos: end of region to score (exclusive). None = kv_len

        Returns:
            scores: Tensor of shape (end_pos - start_pos,),
                    cumulative attention scores per KV position
        """
        if attentions is None or len(attentions) == 0:
            raise ValueError("attentions is None or empty. Cannot compute scores.")

        num_layers = len(attentions)
        # Select last num_score_layers layers
        selected_layers = attentions[max(0, num_layers - self.num_score_layers):]

        kv_len = selected_layers[0].shape[-1]
        if start_pos is None:
            start_pos = 0
        if end_pos is None:
            end_pos = kv_len

        # Clamp end_pos to kv_len to avoid index out of bounds
        end_pos = min(end_pos, kv_len)
        if end_pos <= start_pos:
            return torch.zeros(0, device=selected_layers[0].device)

        all_scores = []
        for attn in selected_layers:
            # attn shape: (batch, num_heads, query_len, kv_len)
            # Sum over heads and query positions -> score per KV position
            # shape: (batch, kv_len) after summing heads and queries
            layer_score = attn[0].sum(dim=0).sum(dim=0)  # (kv_len,)
            all_scores.append(layer_score[start_pos:end_pos])

        # Average across selected layers
        scores = torch.stack(all_scores, dim=0).mean(dim=0)  # (end_pos - start_pos,)
        return scores

    def compute_attention_matrix(self, attentions, key_start=None, key_end=None):
        """
        Average last layers + heads into a (query_len, kv_len) attention matrix.
        """
        if attentions is None or len(attentions) == 0:
            raise ValueError("attentions is None or empty. Cannot compute attention matrix.")

        selected_layers = attentions[max(0, len(attentions) - self.num_score_layers):]
        kv_len = int(selected_layers[0].shape[-1])
        if key_start is None:
            key_start = 0
        if key_end is None:
            key_end = kv_len
        key_start = max(0, int(key_start))
        key_end = min(kv_len, int(key_end))
        if key_end <= key_start:
            return torch.zeros(0, 0, device=selected_layers[0].device)

        mats = []
        for attn in selected_layers:
            layer_mat = attn[0].mean(dim=0)  # (query_len, kv_len)
            mats.append(layer_mat[:, key_start:key_end])
        return torch.stack(mats, dim=0).mean(dim=0)

    def select_heavy_hitters(self, scores, keep_ratio=0.5, min_keep=1):
        """
        H2O Algorithm: Select top-k heavy hitters based on attention scores.
        
        Keep the tokens with the highest cumulative attention scores.
        This is the practical implementation of H2O: evaluate all tokens
        and keep those with highest importance scores globally.
        
        Args:
            scores: Importance scores for each token position (shape: n,)
            keep_ratio: Fraction of tokens to keep (default: 0.5 = keep 50%)
            min_keep: Minimum number of tokens to always keep (default: 1)
        
        Returns:
            (heavy_hitter_indices, evicted_indices): Indices to keep and evict
        """
        device = scores.device
        n = scores.shape[0]
        
        # ========================
        # 1. Calculate target budget k
        # ========================
        k = max(min_keep, int(n * keep_ratio))
        k = min(k, n)
        
        # ========================
        # 2. If no eviction needed
        # ========================
        if n <= k:
            return (
                torch.arange(n, device=device),
                torch.tensor([], dtype=torch.long, device=device),
            )
        
        # ========================
        # 3. H2O: Select top-k by score
        # ========================
        # Get indices of top-k highest scores
        _, top_k_indices = torch.topk(scores, k, dim=0)
        top_k_indices = torch.sort(top_k_indices)[0]  # Sort to maintain order
        
        # ========================
        # 4. Build keep/evict masks
        # ========================
        all_indices = torch.arange(n, device=device)
        
        mask = torch.zeros(n, dtype=torch.bool, device=device)
        mask[top_k_indices] = True
        
        heavy_hitter_indices = all_indices[mask]
        evicted_indices = all_indices[~mask]
        
        return heavy_hitter_indices, evicted_indices