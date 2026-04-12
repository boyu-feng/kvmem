"""
Token Tracking System for H2O KV Cache Pruning

Tracks token indices, pruning events, and cache length changes throughout the episode.
"""

class TokenTracker:
    """Track token positions and pruning history."""
    
    def __init__(self):
        self.token_history = []  # List of (step, event_type, token_range, detail)
        self.current_token_count = 0
        self.pruning_events = []
        
    def add_prefill_tokens(self, step, num_tokens):
        """Record prefill tokens (initial prompt)."""
        start_idx = self.current_token_count
        self.current_token_count += num_tokens
        end_idx = self.current_token_count - 1
        
        event = {
            "step": step,
            "event": "prefill",
            "token_range": f"[{start_idx}:{end_idx}]",
            "num_tokens": num_tokens,
            "total_cache": self.current_token_count
        }
        self.token_history.append(event)
        return event
    
    def add_generated_tokens(self, step, num_tokens):
        """Record newly generated tokens."""
        start_idx = self.current_token_count
        self.current_token_count += num_tokens
        end_idx = self.current_token_count - 1
        
        event = {
            "step": step,
            "event": "generated",
            "token_range": f"[{start_idx}:{end_idx}]",
            "num_tokens": num_tokens,
            "total_cache": self.current_token_count
        }
        self.token_history.append(event)
        return event
    
    def record_pruning(self, step, discarded_token_indices, new_cache_len):
        """Record pruning event with discarded token indices."""
        event = {
            "step": step,
            "event": "pruning",
            "discarded_tokens": discarded_token_indices,
            "num_discarded": len(discarded_token_indices),
            "cache_before": self.current_token_count,
            "cache_after": new_cache_len
        }
        self.pruning_events.append(event)
        self.current_token_count = new_cache_len
        self.token_history.append(event)
        return event
    
    def print_step_summary(self, step, cache_len):
        """Print summary for a step."""
        print(f"\n[TOKEN TRACKING] Step {step}:")
        print(f"  Current cache length: {cache_len}")
        
        # Show recent events
        recent_events = [e for e in self.token_history if e.get("step") == step]
        
        # Separate events by type
        has_pruning = False
        for event in recent_events:
            if event["event"] == "prefill":
                print(f"  ✓ Prefilled tokens {event['token_range']} ({event['num_tokens']} tokens)")
            elif event["event"] == "generated":
                print(f"  ✓ Generated tokens {event['token_range']} ({event['num_tokens']} tokens)")
            elif event["event"] == "pruning":
                has_pruning = True
                num_pruned = event['num_discarded']
                pruned_summary = str(event['discarded_tokens'][:10])
                if len(event['discarded_tokens']) > 10:
                    pruned_summary = pruned_summary[:-1] + ", ...]"
                print(f"  ✗ Pruned {num_pruned} tokens by H2O: {pruned_summary}")
                print(f"    Cache: {event['cache_before']} → {event['cache_after']}")
        
        # If no pruning detected but cache changed, show the cache state
        if not has_pruning:
            print(f"  [INFO] Cache length: {cache_len} (no pruning detected in this step)")
    
    def print_full_history(self):
        """Print complete token tracking history."""
        line = "=" * 90
        print("\n" + line)
        print("TOKEN TRACKING HISTORY".center(90))
        print(line)
        
        for event in self.token_history:
            step = event.get("step", "?")
            if event["event"] == "prefill":
                print(f"[Step {step}] Prefill:    tokens {event['token_range']:15s} ({event['num_tokens']:4d}t) cache: {event['total_cache']:5d}t")
            elif event["event"] == "generated":
                print(f"[Step {step}] Generate:   tokens {event['token_range']:15s} ({event['num_tokens']:4d}t) cache: {event['total_cache']:5d}t")
            elif event["event"] == "pruning":
                discarded = event['discarded_tokens'][:8]
                discarded_str = str(discarded)[:30] + ("..." if len(event['discarded_tokens']) > 8 else "")
                print(f"[Step {step}] Prune:      remove {event['num_discarded']:4d}t {discarded_str:30s} cache: {event['cache_after']:5d}t")
        
        print(line)
    
    def get_statistics(self):
        """Get pruning statistics."""
        total_pruned = sum(e.get("num_discarded", 0) for e in self.pruning_events)
        num_prune_events = len(self.pruning_events)
        
        return {
            "total_pruned_tokens": total_pruned,
            "num_prune_events": num_prune_events,
            "current_cache_length": self.current_token_count,
            "pruning_events": self.pruning_events
        }
