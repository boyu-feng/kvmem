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

    def select_heavy_hitters(self, scores, keep_ratio=0.5, min_keep=1):
        """
        True H2O Algorithm (per pseudocode): Iterative token replacement.
        
        Algorithm:
        1. Initialize S_0 = empty set
        2. For i = 1 to n (each token):
           - If i <= k: Add token i to set (fill budget)
           - Else:
             * Find token u in S_{i-1} union {i} that minimizes score
             * Replace: S_i = (S_{i-1} union {i}) without {u}
        3. Return S_n as kept tokens, others as evicted
        
        Args:
            scores: Importance scores for each token position (shape: n,)
            keep_ratio: Fraction of tokens to keep (default: 0.5)
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
        # 3. True H2O: Iterative replacement
        # ========================
        # Start with empty set
        kept_set = set()  # Indices of kept tokens
        
        for i in range(n):
            if i < k:
                # Phase 1: Fill budget with first k tokens
                kept_set.add(i)
            else:
                # Phase 2: For each new token i, decide whether to keep it
                # by finding the best token to evict
                
                # Candidate set: current kept tokens + new token i
                candidates = kept_set | {i}
                
                best_evict_idx = None
                best_evict_score = float('inf')
                
                # Try evicting each token in the candidate set
                for evict_idx in candidates:
                    # Score of token to evict (lower score = better to evict)
                    loss = scores[evict_idx].item()
                    
                    # Keep track of token with lowest score (best to evict)
                    if loss < best_evict_score:
                        best_evict_score = loss
                        best_evict_idx = evict_idx
                
                # If the best candidate is the new token i, don't add it
                if best_evict_idx == i:
                    # Do nothing, token i is not added
                    pass
                else:
                    # Add new token i, remove token best_evict_idx
                    kept_set.remove(best_evict_idx)
                    kept_set.add(i)
        
        # ========================
        # 4. Convert set to tensors
        # ========================
        kept_indices = torch.tensor(sorted(list(kept_set)), 
                                    dtype=torch.long, device=device)
        
        # All indices that are not kept
        all_indices = torch.arange(n, device=device)
        mask = torch.zeros(n, dtype=torch.bool, device=device)
        mask[kept_indices] = True
        evicted_indices = all_indices[~mask]
        
        return kept_indices, evicted_indices