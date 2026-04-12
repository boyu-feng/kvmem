"""
Token Tracking System for H2O KV Cache Pruning

Tracks which tokens are discarded at each step.
"""

class TokenTracker:
    """Track pruning events per step."""
    
    def __init__(self):
        self.step_pruning_events = {}  # step -> list of discarded indices
        self.total_discarded = 0
        
    def record_token_pruning(self, step, discarded_indices):
        """Record discarded tokens for a step."""
        if step not in self.step_pruning_events:
            self.step_pruning_events[step] = []
        
        # Extend the list with new discarded indices
        self.step_pruning_events[step].extend(discarded_indices)
        self.total_discarded += len(discarded_indices)
    
    def print_step_pruning_summary(self, step):
        """Print pruning summary at the end of a step."""
        if step in self.step_pruning_events and self.step_pruning_events[step]:
            discarded = sorted(set(self.step_pruning_events[step]))  # Remove duplicates and sort
            num_discarded = len(discarded)
            
            # Show first 20 indices and total count
            if len(discarded) > 20:
                indices_str = str(discarded[:20])[:-1] + ", ...]"
            else:
                indices_str = str(discarded)
            
            print(f"[Step {step}] Pruned {num_discarded} tokens: {indices_str}")
        else:
            print(f"[Step {step}] No tokens pruned")
    
    def print_final_summary(self):
        """Print final summary."""
        print(f"\n[FINAL] Total tokens discarded: {self.total_discarded}")
        if self.step_pruning_events:
            print(f"[FINAL] Pruning events per step:")
            for step in sorted(self.step_pruning_events.keys()):
                num = len(set(self.step_pruning_events[step]))
                if num > 0:
                    print(f"  Step {step}: {num} tokens")
