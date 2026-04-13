"""
Token Tracking System for H2O KV Cache Pruning

Tracks which tokens are discarded at each step, using GLOBAL token IDs
that never change even after tokens are pruned.

Key concept:
- Each token gets a GLOBAL ID starting from 0 (the first token in prompt)
- Even if token 5 is deleted, tokens 6, 7, etc. keep their IDs
- This enables consistent tracking across the entire conversation
"""

class TokenTracker:
    """
    Track pruning events per step using global token IDs.
    
    The mapper maintains a list of all global token IDs that still exist in cache:
    - Initial state: [0, 1, 2, 3, ..., initial_length-1]
    - After pruning: [0, 1, 3, 5, 7, ...] (global IDs, not sequential)
    - Cache position maps to global ID: cache[0] has global_id mapper[0]
    """
    
    def __init__(self):
        """Initialize tracker."""
        self.step_pruning_events = {}  # step -> list of discarded GLOBAL token IDs
        self.total_discarded = 0
        self.cache_length = 0
        
        # Mapper: list of global token IDs currently in cache
        # After each prune, this list shrinks but contains original global IDs
        self.global_id_mapper = []
        
    def set_initial_cache_length(self, initial_len):
        """
        Set the initial cache length (e.g., after encoding system prompt + question).
        Maps all positions to their global IDs [0, 1, 2, ..., initial_len-1].
        
        Args:
            initial_len: Initial cache length
        """
        self.cache_length = initial_len
        self.global_id_mapper = list(range(initial_len))
    
    def record_pruning_with_kept_indices(self, step, kept_local_indices, old_cache_length):
        """
        Record pruning by specifying which local indices were KEPT.
        
        Args:
            step: Step number
            kept_local_indices: Local indices in the cache that were kept (e.g., [0, 1, 3, 5])
            old_cache_length: Cache length before pruning
        """
        # Keep mapper aligned with actual cache length.
        # In token-level decoding, cache grows continuously and this tracker
        # may be called without explicit append updates.
        if len(self.global_id_mapper) < old_cache_length:
            start = len(self.global_id_mapper)
            self.global_id_mapper.extend(range(start, old_cache_length))

        # Identify discarded local indices
        all_local_indices = set(range(old_cache_length))
        kept_set = set(kept_local_indices)
        discarded_local_indices = list(all_local_indices - kept_set)
        
        # Convert local indices to global IDs
        discarded_global_ids = [self.global_id_mapper[i] for i in sorted(discarded_local_indices)]
        
        # Update mapper to only keep the kept indices
        new_mapper = [self.global_id_mapper[i] for i in sorted(kept_local_indices)]
        self.global_id_mapper = new_mapper
        self.cache_length = len(self.global_id_mapper)
        
        # Record the event
        if step not in self.step_pruning_events:
            self.step_pruning_events[step] = []
        self.step_pruning_events[step].extend(discarded_global_ids)
        self.total_discarded += len(discarded_global_ids)

    def print_step_pruning_summary(self, step):
        """
        Print pruning summary at the end of a step.
        Shows: cache length and global token IDs that were deleted.
        """
        print(f"[Step {step}] KV cache length: {self.cache_length}")
        
        if step in self.step_pruning_events and self.step_pruning_events[step]:
            discarded = sorted(set(self.step_pruning_events[step]))  # Remove duplicates and sort
            num_discarded = len(discarded)
            
            # Show first 20 indices and total count
            if len(discarded) > 20:
                indices_str = str(discarded[:20])[:-1] + ", ...]"
            else:
                indices_str = str(discarded)
            
            print(f"[Step {step}] Discarded {num_discarded} tokens: {indices_str}")
        else:
            print(f"[Step {step}] No tokens discarded")
    
    def print_final_summary(self):
        """Print final summary with total statistics."""
        print(f"\n[FINAL] Total tokens discarded: {self.total_discarded}")
        print(f"[FINAL] Final cache length: {self.cache_length}")
        if self.step_pruning_events:
            print(f"[FINAL] Pruning events per step:")
            for step in sorted(self.step_pruning_events.keys()):
                num = len(set(self.step_pruning_events[step]))
                if num > 0:
                    print(f"  Step {step}: {num} tokens")
