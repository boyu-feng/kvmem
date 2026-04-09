#!/usr/bin/env python3
"""
Simple test for _SimpleDynamicCache.update method
"""

import torch
import sys
sys.path.insert(0, '/Users/fengboyu/Documents/Python_Code/kvmem')

# Simulate the _SimpleDynamicCache classes
class _SimpleLayer:
    def __init__(self, k, v):
        self.keys = k.detach().clone()
        self.values = v.detach().clone()

class _SimpleDynamicCache:
    def __init__(self, kv_tuple):
        # kv_tuple: tuple of (k, v) per layer
        self.layers = []
        for k, v in kv_tuple:
            self.layers.append(_SimpleLayer(k, v))

    def get_seq_length(self):
        if not self.layers:
            return 0
        return self.layers[0].keys.shape[2]

    def crop(self, keep_token_count):
        for layer in self.layers:
            layer.keys = layer.keys[:, :, :keep_token_count, :].detach().clone()
            layer.values = layer.values[:, :, :keep_token_count, :].detach().clone()

    def get_mask_sizes(self, cache_position=None, layer_idx=None):
        """
        Compatibility shim for transformers' masking utilities.
        Returns (kv_length, kv_offset) where kv_length is the length of
        the cached key/value sequence and kv_offset is the offset
        (we return 0 since this simple cache uses absolute positions).
        """
        kv_length = self.get_seq_length()
        kv_offset = 0
        return kv_length, kv_offset
    
    @property
    def device(self):
        # return device of underlying tensors if available
        if not self.layers:
            return None
        return self.layers[0].keys.device

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        """
        Update the cache with new key and value states for a specific layer.
        This method is called by the model during forward pass.
        
        Args:
            key_states: new key tensor to append
            value_states: new value tensor to append
            layer_idx: index of the layer being updated
            cache_kwargs: optional kwargs (unused in this simple implementation)
        
        Returns:
            Tuple of (updated_key_states, updated_value_states)
        """
        # Ensure layer index is valid
        while len(self.layers) <= layer_idx:
            self.layers.append(_SimpleLayer(
                torch.empty((1, 1, 0, key_states.shape[-1]), 
                          dtype=key_states.dtype, device=key_states.device),
                torch.empty((1, 1, 0, value_states.shape[-1]), 
                          dtype=value_states.dtype, device=value_states.device)
            ))
        
        # Concatenate new key/value with existing cache
        if self.layers[layer_idx].keys.shape[2] == 0:
            # Empty cache, initialize with new states
            self.layers[layer_idx].keys = key_states.detach().clone()
            self.layers[layer_idx].values = value_states.detach().clone()
        else:
            # Append to existing cache
            self.layers[layer_idx].keys = torch.cat(
                [self.layers[layer_idx].keys, key_states], dim=2
            ).detach().clone()
            self.layers[layer_idx].values = torch.cat(
                [self.layers[layer_idx].values, value_states], dim=2
            ).detach().clone()
        
        # Return full cached key/value (what the model expects)
        return self.layers[layer_idx].keys, self.layers[layer_idx].values


def test_simple_dynamic_cache():
    """Test the _SimpleDynamicCache class"""
    print("=" * 60)
    print("Testing _SimpleDynamicCache")
    print("=" * 60)
    
    # Create initial cache: 3 layers, each with 10 tokens, head_dim=128
    batch_size, num_heads, seq_len, head_dim = 1, 4, 10, 128
    num_layers = 3
    
    initial_kv = []
    for layer_idx in range(num_layers):
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)
        initial_kv.append((k, v))
    
    # Create cache
    cache = _SimpleDynamicCache(tuple(initial_kv))
    print(f"✓ Created cache with {len(cache.layers)} layers")
    print(f"  Initial seq_length: {cache.get_seq_length()}")
    
    # Test update method for each layer
    new_seq_len = 5
    for layer_idx in range(num_layers):
        new_k = torch.randn(batch_size, num_heads, new_seq_len, head_dim, dtype=torch.float32)
        new_v = torch.randn(batch_size, num_heads, new_seq_len, head_dim, dtype=torch.float32)
        
        # Call update
        updated_k, updated_v = cache.update(new_k, new_v, layer_idx)
        
        # Verify
        expected_len = seq_len + new_seq_len
        actual_len = updated_k.shape[2]
        assert actual_len == expected_len, f"Layer {layer_idx}: expected len {expected_len}, got {actual_len}"
        print(f"✓ Layer {layer_idx}: update succeeded. New seq_length: {actual_len}")
    
    # Test get_mask_sizes
    kv_length, kv_offset = cache.get_mask_sizes()
    print(f"✓ get_mask_sizes: kv_length={kv_length}, kv_offset={kv_offset}")
    
    # Test device property
    device = cache.device
    print(f"✓ device property: {device}")
    
    # Test auto-extending layers
    cache2 = _SimpleDynamicCache(initial_kv[:1])  # Start with 1 layer
    new_k = torch.randn(batch_size, num_heads, new_seq_len, head_dim, dtype=torch.float32)
    new_v = torch.randn(batch_size, num_heads, new_seq_len, head_dim, dtype=torch.float32)
    
    # Update layer 5 (doesn't exist yet)
    updated_k, updated_v = cache2.update(new_k, new_v, layer_idx=5)
    print(f"✓ Auto-extend layers: cache now has {len(cache2.layers)} layers")
    
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_simple_dynamic_cache()
