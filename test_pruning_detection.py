#!/usr/bin/env python3
"""
Test script to verify pruning detection in token tracking.
"""

from token_tracker import TokenTracker

# Simulate a multi-step episode
tracker = TokenTracker()

print("=" * 80)
print("SIMULATING TOKEN TRACKING WITH PRUNING DETECTION")
print("=" * 80)

# Step 1: Initial prefill + generation
print("\n[Step 1] Initial Prefill + Generation:")
print("  Initial prompt: 1674 tokens [0:1673]")
print("  Generated: 40 tokens [1674:1713]")

tracker.add_prefill_tokens(1, 1674)
tracker.add_generated_tokens(1, 40)
tracker.print_step_summary(1, 1714)

# Step 2: Prefill + Generation + Pruning
print("\n[Step 2] Prefill + Generation + Pruning:")
print("  New observation: 65 tokens")
print("  Generated: 32 tokens")
print("  PRUNING: 267 tokens removed by H2O")
print("  Cache before: 1811, Cache after: 1544")

tracker.add_prefill_tokens(2, 65)
tracker.add_generated_tokens(2, 32)
# Simulate pruning detection
pruned_indices = list(range(420, 687))  # Simulated pruned token range
tracker.record_pruning(2, pruned_indices, 1544)
tracker.print_step_summary(2, 1544)

# Step 3: Prefill + Generation
print("\n[Step 3] Prefill + Generation (No Pruning):")
print("  New observation: 61 tokens")
print("  Generated: 31 tokens")
print("  Cache before: 1605, Cache after: 1636")

tracker.add_prefill_tokens(3, 61)
tracker.add_generated_tokens(3, 31)
tracker.print_step_summary(3, 1636)

# Step 4: Prefill + Generation + Pruning
print("\n[Step 4] Prefill + Generation + Pruning:")
print("  New observation: 58 tokens")
print("  Generated: 29 tokens")
print("  PRUNING: 189 tokens removed by H2O")
print("  Cache before: 1723, Cache after: 1534")

tracker.add_prefill_tokens(4, 58)
tracker.add_generated_tokens(4, 29)
pruned_indices = list(range(512, 701))
tracker.record_pruning(4, pruned_indices, 1534)
tracker.print_step_summary(4, 1534)

# Print final summary
print("\n")
tracker.print_full_history()

# Print statistics
stats = tracker.get_statistics()
print(f"\n[TOKEN STATS] Total pruned: {stats['total_pruned_tokens']}, "
      f"Prune events: {stats['num_prune_events']}, "
      f"Final cache: {stats['current_cache_length']}")
