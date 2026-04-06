"""
Qwen LLM Wrapper with KV Cache Management
Supports incremental generation with KV cache reuse and pruning.
Provides two attention acquisition modes:
  1. Scoring forward: dedicated forward pass with output_attentions=True at pruning time
  2. Piggyback: record attention during normal prefill of new observation tokens
"""

import torch
import time
import copy
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from kv_cache.kv_cache_manager import KVCacheManager


class QwenLLMWithKVCache:
    """
    Qwen LLM wrapper with KV Cache management for incremental generation.

    Instead of re-encoding the full trajectory every step, this wrapper:
    1. Encodes [System + Question] once, caching KV
    2. For each new step, only encodes new tokens (Observation), reusing cached KV
    3. Optionally prunes the KV cache to control memory
    """

    def __init__(self, model_path, kv_config=None):
        """
        Args:
            model_path: path to the pretrained Qwen model
            kv_config: dict with pruning configuration (see KVCacheManager).
                       If None, no pruning is applied (pure KV cache reuse).
        """
        print(f"[INFO] Loading model from {model_path} (KV Cache mode)...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

        # Always load with default (SDPA) attention for best inference quality.
        # For H2O scoring, we temporarily switch to eager attention only during
        # the scoring forward pass (output_attentions=True requires eager).
        pruning_mode = (kv_config or {}).get("pruning_mode", "none")
        self.needs_attn_scoring = pruning_mode in ("h2o", "h2o_snapkv")
        self.needs_new_step_kv = pruning_mode in ("ours")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        self.device = self.model.device
        attn_impl = getattr(self.model.config, '_attn_implementation', 'default')
        print(f"[INFO] Model loaded successfully (KV Cache mode). attn_implementation={attn_impl}")
        if self.needs_attn_scoring:
            print("[INFO] H2O scoring will temporarily switch to eager attention during scoring forward.")
        if self.needs_new_step_kv:
            print("[INFO] Ours pruning will require new step KV during fusion.")

        # KV Cache management
        self.kv_config = kv_config or {}
        self.pruning_enabled = self.kv_config.get("pruning_mode", "none") != "none"
        if self.pruning_enabled:
            self.kv_manager = KVCacheManager(self.kv_config)
        else:
            self.kv_manager = None

        # Attention acquisition mode: "scoring_forward" or "piggyback"
        self.attn_mode = self.kv_config.get("attn_mode", "scoring_forward")
        self.step_kv = None
        # State
        self.past_key_values = None
        self.current_cache_len = 0

        # Timing stats
        self.timing_stats = {
            "prefill_time": 0.0,
            "decode_time": 0.0,
            "scoring_time": 0.0,
            "pruning_time": 0.0,
        }

    def reset(self):
        """Reset all state for a new episode."""
        self.past_key_values = None
        self.current_cache_len = 0
        self._all_token_ids = []  # Track all token ids seen for repetition penalty
        if self.kv_manager:
            self.kv_manager.register_initial_cache(0)
        self.timing_stats = {
            "prefill_time": 0.0,
            "decode_time": 0.0,
            "scoring_time": 0.0,
            "pruning_time": 0.0,
        }

    def truncate_cache(self, keep_token_count):
        """
        Truncate the KV cache to keep only the first `keep_token_count` tokens.
        
        This is CRITICAL for correctness: after generation, the model may have 
        generated tokens beyond the useful Action line (e.g., hallucinated 
        Observation/Thought/Action for future steps). These hallucinated tokens
        must be removed from the KV cache before appending the real Observation,
        otherwise the model sees conflicting context.
        
        Args:
            keep_token_count: number of tokens to keep from the start of the cache
        """
        if self.past_key_values is None or keep_token_count >= self.current_cache_len:
            return
        
        if isinstance(self.past_key_values, DynamicCache):
            # Use DynamicCache's built-in crop method
            self.past_key_values.crop(keep_token_count)
        else:
            # Tuple of (key, value) pairs
            truncated = []
            for k, v in self.past_key_values:
                truncated.append((k[:, :, :keep_token_count, :], v[:, :, :keep_token_count, :]))
            self.past_key_values = tuple(truncated)
        
        old_len = self.current_cache_len
        self.current_cache_len = keep_token_count
        
        # Also truncate tracked token ids
        if len(self._all_token_ids) > keep_token_count:
            self._all_token_ids = self._all_token_ids[:keep_token_count]
        
        # Update manager
        if self.kv_manager:
            self.kv_manager.current_cache_len = self.current_cache_len

    def generate_first(self, prompt_text, max_new_tokens=256, stop_strings=None):
        """
        First call: encode the full initial prompt, cache KV, and generate response.
        """
        self.reset()
    
        # Tokenize the full prompt
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_text},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        prompt_len = input_ids.shape[1]
    
        # Use model.generate()
        t0 = time.time()
        with torch.no_grad():
            gen_outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                return_dict_in_generate=True,
                use_cache=True,
            )
        total_time = time.time() - t0
    
        # Extract generated token IDs
        generated_ids = gen_outputs.sequences[0][prompt_len:].tolist()
    
        # Extract past_key_values
        if hasattr(gen_outputs, 'past_key_values') and gen_outputs.past_key_values is not None:
            self.past_key_values = gen_outputs.past_key_values
        else:
            print("[WARN] model.generate() did not return past_key_values. Running fallback prefill.")
            all_ids = gen_outputs.sequences[:, :prompt_len + len(generated_ids)]
            with torch.no_grad():
                outputs = self.model(
                    input_ids=all_ids,
                    use_cache=True,
                    return_dict=True,
                )
            self.past_key_values = outputs.past_key_values
    
        self.current_cache_len = prompt_len + len(generated_ids)
    
        # 提取 KV caches - 正确处理 DynamicCache
        prompt_kv = []
        generated_kv = []
        
        # 打印调试信息
        print(f"[DEBUG] past_key_values type: {type(self.past_key_values)}")
        print(f"[DEBUG] past_key_values attributes: {[attr for attr in dir(self.past_key_values) if not attr.startswith('_')]}")
        
        # 方法1: 如果是 DynamicCache，尝试遍历 layers
        if hasattr(self.past_key_values, 'layers'):
            print(f"[DEBUG] Using layers attribute, number of layers: {len(self.past_key_values.layers)}")
            for layer_idx, layer in enumerate(self.past_key_values.layers):
                print(f"[DEBUG] Layer {layer_idx} type: {type(layer)}")
                # 每个 layer 可能是 DynamicLayer 对象
                if hasattr(layer, 'key') and hasattr(layer, 'value'):
                    k = layer.key
                    v = layer.value
                elif hasattr(layer, 'k') and hasattr(layer, 'v'):
                    k = layer.k
                    v = layer.v
                elif isinstance(layer, tuple) and len(layer) == 2:
                    k, v = layer
                else:
                    print(f"[ERROR] Cannot unpack layer {layer_idx}: {layer}")
                    continue
                
                # 提取 Prompt 部分
                pk = k[:, :, :prompt_len, :].clone()
                pv = v[:, :, :prompt_len, :].clone()
                prompt_kv.append((pk, pv))
                
                # 提取生成部分
                gk = k[:, :, prompt_len:, :].clone()
                gv = v[:, :, prompt_len:, :].clone()
                generated_kv.append((gk, gv))
        
        # 方法2: 使用 key_cache 和 value_cache (旧版 DynamicCache)
        elif hasattr(self.past_key_values, 'key_cache') and hasattr(self.past_key_values, 'value_cache'):
            print(f"[DEBUG] Using key_cache/value_cache, number of layers: {len(self.past_key_values.key_cache)}")
            for layer_idx in range(len(self.past_key_values.key_cache)):
                k = self.past_key_values.key_cache[layer_idx]
                v = self.past_key_values.value_cache[layer_idx]
                
                pk = k[:, :, :prompt_len, :].clone()
                pv = v[:, :, :prompt_len, :].clone()
                prompt_kv.append((pk, pv))
                
                gk = k[:, :, prompt_len:, :].clone()
                gv = v[:, :, prompt_len:, :].clone()
                generated_kv.append((gk, gv))
        
        # 方法3: 尝试转换为 tuple
        elif hasattr(self.past_key_values, 'to_tuple'):
            print(f"[DEBUG] Using to_tuple() method")
            pkv_tuple = self.past_key_values.to_tuple()
            for layer_idx in range(len(pkv_tuple)):
                k, v = pkv_tuple[layer_idx]
                
                pk = k[:, :, :prompt_len, :].clone()
                pv = v[:, :, :prompt_len, :].clone()
                prompt_kv.append((pk, pv))
                
                gk = k[:, :, prompt_len:, :].clone()
                gv = v[:, :, prompt_len:, :].clone()
                generated_kv.append((gk, gv))
        
        # 方法4: 直接作为序列处理
        elif isinstance(self.past_key_values, (list, tuple)):
            print(f"[DEBUG] Using list/tuple access")
            for layer_idx in range(len(self.past_key_values)):
                k, v = self.past_key_values[layer_idx]
                
                pk = k[:, :, :prompt_len, :].clone()
                pv = v[:, :, :prompt_len, :].clone()
                prompt_kv.append((pk, pv))
                
                gk = k[:, :, prompt_len:, :].clone()
                gv = v[:, :, prompt_len:, :].clone()
                generated_kv.append((gk, gv))
        
        else:
            print(f"[ERROR] Cannot extract KV cache from type: {type(self.past_key_values)}")
            print(f"[ERROR] Available methods: {dir(self.past_key_values)}")
            # 创建空的 KV 作为 fallback
            num_layers = self.model.config.num_hidden_layers
            for _ in range(num_layers):
                empty_k = torch.zeros(1, self.model.config.num_attention_heads, 0, self.model.config.head_dim).to(self.device)
                empty_v = torch.zeros(1, self.model.config.num_attention_heads, 0, self.model.config.head_dim).to(self.device)
                prompt_kv.append((empty_k, empty_v))
                generated_kv.append((empty_k, empty_v))
        
        # 转换为元组
        prompt_kv = tuple(prompt_kv)
        generated_kv = tuple(generated_kv)
        
        print(f"[DEBUG] Final prompt_kv length: {len(prompt_kv)}")
        print(f"[DEBUG] Final generated_kv length: {len(generated_kv)}")
        if len(prompt_kv) > 0:
            print(f"[DEBUG] Sample prompt_kv[0][0] shape: {prompt_kv[0][0].shape}")
    
        # Approximate timing split
        self.timing_stats["prefill_time"] += total_time * 0.3
        self.timing_stats["decode_time"] += total_time * 0.7
    
        # Register initial cache with manager
        if self.kv_manager:
            self.kv_manager.register_initial_cache(prompt_len)
            self.kv_manager.current_cache_len = self.current_cache_len
    
        # Track all token ids
        self._all_token_ids = input_ids[0].tolist() + generated_ids
    
        response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Truncate at stop_strings if found
        if stop_strings:
            truncated_text = response_text
            for stop_str in stop_strings:
                idx = truncated_text.find(stop_str)
                if idx != -1:
                    truncated_text = truncated_text[:idx]
            
            if len(truncated_text) < len(response_text):
                truncated_ids = self.tokenizer(
                    truncated_text, add_special_tokens=False
                ).input_ids
                keep_count = prompt_len + len(truncated_ids)
                self.truncate_cache(keep_count)
                response_text = truncated_text
        
        return response_text.strip(), prompt_kv, generated_kv


    def generate_incremental_with_memory(self, new_text, prompt_kv, memory_block, recent_kv, max_new_tokens=256, stop_strings=None):
        """
        组合 Prompt, Memory 和 Recent KV，进行增量解码，并返回分离的增量 KV。
        """
        # 添加输入验证
        if prompt_kv is None or len(prompt_kv) == 0:
            print("[ERROR] prompt_kv is None or empty, cannot proceed")
            return "", None, None
        
        if memory_block is None or len(memory_block) == 0:
            print("[ERROR] memory_block is None or empty, cannot proceed")
            return "", None, None
        
        # 1. 将 text 转化为 input_ids
        new_input_ids = self.tokenizer(new_text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        new_token_count = new_input_ids.shape[1]
        
        if new_token_count == 0:
            print("[WARN] new_text is empty, skipping prefill")
            return "", None, None
    
        # 2. 物理拼接或逻辑组合 KV Cache
        # 这一步非常关键：Transformer 期待的 past_key_values 是一个元组，每层格式为 (layer_k, layer_v)
        combined_pkv = []
        num_layers = len(prompt_kv)
        
        for i in range(num_layers):
            p_k, p_v = prompt_kv[i]
            m_k, m_v = memory_block[i]
            
            # 确保维度匹配
            # 基础拼接：在 seq_len 维度 (维度 2) 上做 Concat
            # 如果你的 memory_block 依然是原始矩阵形状，直接 concat。
            # 如果它是经过 Delta Rule 压缩后的低秩矩阵，你可能需要先做某种变换或采用 Linear Attention 算子
            layer_k = torch.cat([p_k, m_k], dim=2)
            layer_v = torch.cat([p_v, m_v], dim=2)
            
            if recent_kv is not None and len(recent_kv) > i:
                r_k, r_v = recent_kv[i]
                layer_k = torch.cat([layer_k, r_k], dim=2)
                layer_v = torch.cat([layer_v, r_v], dim=2)
                
            combined_pkv.append((layer_k, layer_v))
            
        combined_pkv = tuple(combined_pkv)
    
        # 3. Prefill 新的 Observation
        with torch.no_grad():
            outputs = self.model(
                input_ids=new_input_ids,
                past_key_values=combined_pkv,
                use_cache=True,
                return_dict=True
            )
            
        # 4. 提取当前 Observation 的增量 KV
        obs_kv = []
        if outputs.past_key_values is not None:
            for layer_pkv in outputs.past_key_values:
                # 处理 DynamicCache 或 tuple
                if hasattr(layer_pkv, 'key_cache') and hasattr(layer_pkv, 'value_cache'):
                    # 如果是 DynamicCache 的层（通常不会，但为了安全）
                    k = layer_pkv.key_cache
                    v = layer_pkv.value_cache
                else:
                    # 标准 tuple 格式
                    k, v = layer_pkv
                
                s_k = k[:, :, -new_token_count:, :].detach().clone()
                s_v = v[:, :, -new_token_count:, :].detach().clone()
                obs_kv.append((s_k, s_v))
        obs_kv = tuple(obs_kv)
    
        # 5. Decode 下一轮的 Thought/Action
        # 这里直接调用你现有的 _decode 方法（它应该基于更新后的 outputs.past_key_values 进行自回归）
        self.past_key_values = outputs.past_key_values
        response_text, generated_len = self._decode(outputs.logits, max_new_tokens)
    
        # 6. 提取模型生成的 Thought/Action 的增量 KV
        gen_kv = []
        if generated_len > 0 and self.past_key_values is not None:
            for layer_pkv in self.past_key_values:
                # 处理 DynamicCache 或 tuple
                if hasattr(layer_pkv, 'key_cache') and hasattr(layer_pkv, 'value_cache'):
                    k = layer_pkv.key_cache
                    v = layer_pkv.value_cache
                else:
                    k, v = layer_pkv
                
                g_k = k[:, :, -generated_len:, :].detach().clone()
                g_v = v[:, :, -generated_len:, :].detach().clone()
                gen_kv.append((g_k, g_v))
        gen_kv = tuple(gen_kv)
    
        # 7. Truncation 逻辑（保留你代码里的 stop_strings 切割逻辑）
        if stop_strings and response_text:
            truncated_text = response_text
            for stop_str in stop_strings:
                idx = truncated_text.find(stop_str)
                if idx != -1:
                    truncated_text = truncated_text[:idx]
            
            if len(truncated_text) < len(response_text):
                # 重新 tokenize 截断后的文本
                truncated_ids = self.tokenizer(
                    truncated_text, add_special_tokens=False
                ).input_ids
                keep_len = len(truncated_ids)
                
                # 截断 KV cache
                if keep_len < generated_len and self.past_key_values is not None:
                    # 截断生成的 KV 部分
                    new_past_kv = []
                    for layer_pkv in self.past_key_values:
                        if hasattr(layer_pkv, 'key_cache') and hasattr(layer_pkv, 'value_cache'):
                            k = layer_pkv.key_cache
                            v = layer_pkv.value_cache
                        else:
                            k, v = layer_pkv
                        
                        # 保留生成的 KV 中前 keep_len 个 token
                        new_k = k[:, :, :-(generated_len - keep_len), :] if generated_len > keep_len else k
                        new_v = v[:, :, :-(generated_len - keep_len), :] if generated_len > keep_len else v
                        new_past_kv.append((new_k, new_v))
                    self.past_key_values = tuple(new_past_kv)
                
                response_text = truncated_text
        
        return response_text.strip(), obs_kv, gen_kv

    def fuse_memory(self, memory_block, new_kv):
        """
        Fuse new KV into memory block using delta rule or simple averaging.
        """
        # 添加检查
        if memory_block is None or len(memory_block) == 0:
            # 如果 memory_block 为空，直接返回 new_kv
            return new_kv
        
        if new_kv is None or len(new_kv) == 0:
            return memory_block
        
        fused_memory = []
        for i in range(len(memory_block)):
            m_k, m_v = memory_block[i]
            n_k, n_v = new_kv[i]
            
            # 这里实现你的融合逻辑
            # 简单的拼接示例：
            fused_k = torch.cat([m_k, n_k], dim=2)
            fused_v = torch.cat([m_v, n_v], dim=2)
            fused_memory.append((fused_k, fused_v))
        
        return tuple(fused_memory)
    
    def generate_incremental(self, new_text, max_new_tokens=256, stop_strings=None):
        """
        Modified version to separate and return Observation KV and Generated KV.
        """
        assert self.past_key_values is not None, \
            "Must call generate_first() before generate_incremental()"

        wrapped_text = new_text
        new_input_ids = self.tokenizer(
            wrapped_text, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(self.device)
        new_token_count = new_input_ids.shape[1]

        # --- 阶段 1: Prefill Observation ---
        t0 = time.time()
        need_attention = (self.pruning_enabled and 
                          self.attn_mode == "piggyback" and 
                          self.kv_config.get("pruning_mode") != "snapkv")

        with torch.no_grad():
            outputs = self.model(
                input_ids=new_input_ids,
                past_key_values=self.past_key_values,
                use_cache=True,
                return_dict=True,
                output_attentions=need_attention,
            )
        
        # 提取这一轮 Observation 的 KV (step_kv)
        obs_kv = []
        for layer_pkv in outputs.past_key_values:
            k, v = layer_pkv
            # 截取最后 new_token_count 个位置
            s_k = k[:, :, -new_token_count:, :].detach().clone()
            s_v = v[:, :, -new_token_count:, :].detach().clone()
            obs_kv.append((s_k, s_v))
        obs_kv = tuple(obs_kv)

        self.past_key_values = outputs.past_key_values
        self.current_cache_len += new_token_count
        self.timing_stats["prefill_time"] += (time.time() - t0)

        # 这里的 step_kv 用于 manager 的剪枝决策（保持原逻辑）
        if self.kv_manager:
            self.kv_manager.append_step(new_token_count)
            self.kv_manager.current_cache_len = self.current_cache_len
            if self.kv_manager.should_prune():
                self._do_pruning(outputs.attentions if need_attention else None, 
                                 step_kv=obs_kv, step_token_count=new_token_count)

        self._all_token_ids.extend(new_input_ids[0].tolist())
        cache_len_before_decode = self.current_cache_len

        # --- 阶段 2: Decode Thought & Action ---
        # 假设 _decode 内部会更新 self.past_key_values 并返回生成长度
        response_text, generated_len = self._decode(
            outputs.logits, max_new_tokens
        )

        # 提取模型生成的 Thought/Action 的 KV (gen_kv)
        gen_kv = []
        if generated_len > 0:
            for layer_pkv in self.past_key_values:
                k, v = layer_pkv
                # 此时 cache_len 已经增加，截取最后 generated_len 个位置
                g_k = k[:, :, -generated_len:, :].detach().clone()
                g_v = v[:, :, -generated_len:, :].detach().clone()
                gen_kv.append((g_k, g_v))
        gen_kv = tuple(gen_kv)

        # --- 阶段 3: Truncation (如果命中 stop_strings) ---
        if stop_strings and response_text:
            truncated_text = response_text
            for stop_str in stop_strings:
                idx = truncated_text.find(stop_str)
                if idx != -1:
                    truncated_text = truncated_text[:idx]
            
            if len(truncated_text) < len(response_text):
                truncated_ids = self.tokenizer(truncated_text, add_special_tokens=False).input_ids
                keep_count = cache_len_before_decode + len(truncated_ids)
                self.truncate_cache(keep_count)
                response_text = truncated_text
                # 注意：如果发生了截断，你可能需要根据 keep_count 重新切分 gen_kv，
                # 这里为了简洁保持原始生成的 gen_kv，或根据业务需求调整。

        # 返回生成文本，以及分开的 KV 元组
        return response_text.strip() if response_text else response_text, obs_kv, gen_kv
    

    def _do_pruning(self, piggyback_attentions=None, step_kv=None, step_token_count=None):
        """
        执行 KV cache 剪枝/压缩。
        
        Args:
            piggyback_attentions: 预填充阶段捕获的注意力权重
            step_kv: 本次 Observation 新增的 KV 矩阵 (元组形式)
            step_token_count: 本次新增 Token 的数量
        """
        attentions = None
        mode = self.kv_config.get("pruning_mode", "h2o_snapkv")

        # --- 1. 获取注意力权重 (原有逻辑保留) ---
        if mode != "snapkv":
            if self.attn_mode == "scoring_forward":
                t0 = time.time()
                attentions = self._get_attention_for_scoring()
                self.timing_stats["scoring_time"] += time.time() - t0
            elif self.attn_mode == "piggyback" and piggyback_attentions is not None:
                attentions = piggyback_attentions
            else:
                t0 = time.time()
                attentions = self._get_attention_for_scoring()
                self.timing_stats["scoring_time"] += time.time() - t0

        # --- 2. 同步状态 ---
        self.kv_manager.current_cache_len = self.current_cache_len

        # --- 3. 调用核心算法 ---
        t0 = time.time()
        
        # 注意：这里的 self.past_key_values 是包含 step_kv 的全量 KV
        # 你的 prune 算法内部会根据 step_token_count 再次划分 Base 和 Signal
        new_kv, new_len, info = self.kv_manager.prune(
            past_key_values=self.past_key_values, 
            attentions=attentions,
            step_kv=step_kv,                # 明确传入新增信号 S
            step_token_count=step_token_count 
        )
        
        # --- 4. 更新状态 ---
        self.past_key_values = new_kv
        self.current_cache_len = new_len
        self.timing_stats["pruning_time"] += time.time() - t0
        
        # 记录剪枝历史（可选）
        if hasattr(self, 'pruning_history'):
            self.pruning_history.append(info)

    def _get_attention_for_scoring(self):
        """
        Run a forward pass with output_attentions=True to get attention weights.

        Uses the last token in the cache as the query.
        Attention shape per layer: (1, num_heads, 1, cache_len+1),
        so memory overhead is only O(num_layers * num_heads * cache_len).

        IMPORTANT: We must NOT update self.past_key_values with the dummy token.
        We deep-copy the cache before the forward pass so the original is untouched.

        We temporarily switch to eager attention for this pass since SDPA
        does not support output_attentions=True.
        """
        dummy_input = torch.tensor([[self.tokenizer.eos_token_id]], device=self.device)

        # Deep copy the cache so the dummy token doesn't pollute it
        if isinstance(self.past_key_values, DynamicCache):
            cache_copy = copy.deepcopy(self.past_key_values)
        else:
            cache_copy = tuple(
                (k.clone(), v.clone()) for k, v in self.past_key_values
            )

        try:
            # Temporarily switch to eager attention for output_attentions support.
            # All attention layers read from self.config._attn_implementation dynamically
            # in their forward(), so we only need to change the config object.
            original_impl = getattr(self.model.config, '_attn_implementation', None)
            self.model.config._attn_implementation = 'eager'

            with torch.no_grad():
                outputs = self.model(
                    input_ids=dummy_input,
                    past_key_values=cache_copy,
                    use_cache=True,
                    return_dict=True,
                    output_attentions=True,
                )

            # Restore original attention implementation
            self.model.config._attn_implementation = original_impl

            attentions = outputs.attentions
            if attentions is None or len(attentions) == 0:
                print("[WARN] output_attentions=True returned None even with eager attention.")
                return None
            return attentions
        except Exception as e:
            # Restore on failure
            try:
                self.model.config._attn_implementation = original_impl
            except Exception:
                pass
            print(f"[WARN] Scoring forward failed: {e}")
            return None

    def _decode(self, last_logits, max_new_tokens):
        """
        Decode using model.generate() with past_key_values for efficient generation.

        After the caller has done a manual prefill (encoding new observation tokens
        into the KV cache), this method handles the autoregressive decode phase
        using model.generate() — which is much faster than a manual Python
        token-by-token loop.

        Strategy:
        - Get the first decoded token from last_logits (argmax)
        - Pass this single token to model.generate() along with past_key_values
          AND an explicit cache_position to avoid HF's internal cache_position
          computation bug (which produces an empty tensor when past_key_values
          length exceeds input_ids length)
        - model.generate() processes this token (adding it to KV cache) then
          continues decoding efficiently using its optimized internal loop
        - The first token in the output IS the token we passed in, so the full
          generated sequence is returned correctly

        Args:
            last_logits: logits from the prefill step, shape (1, seq_len, vocab_size).
                         We use the last position's logits to get the first decode token.
            max_new_tokens: maximum tokens to generate

        Returns:
            response_text: decoded string
            generated_len: number of tokens generated
        """
        t0 = time.time()

        # Get the first decode token from the prefill logits
        first_token_logits = last_logits[:, -1, :]  # (1, vocab_size)
        first_token_id = first_token_logits.argmax(dim=-1, keepdim=True)  # (1, 1)

        # Check if the first token is already an EOS token
        gen_config = getattr(self.model, 'generation_config', None)
        if gen_config and gen_config.eos_token_id is not None:
            eos_ids = gen_config.eos_token_id
            if isinstance(eos_ids, int):
                eos_ids = {eos_ids}
            else:
                eos_ids = set(eos_ids)
        else:
            eos_ids = {self.tokenizer.eos_token_id}

        if first_token_id.item() in eos_ids:
            # Model wants to stop immediately
            self.timing_stats["decode_time"] += time.time() - t0
            return "", 0

        # Use model.generate() with the first token + past_key_values.
        # generate() will:
        # 1. Process first_token_id (1 token prefill, adding it to KV cache)
        # 2. Run optimized autoregressive decode for remaining tokens
        #
        # We need attention_mask covering: existing KV cache + this 1 new token
        attention_mask = torch.ones(
            (1, self.current_cache_len + 1), dtype=torch.long, device=self.device
        )

        # CRITICAL: Provide cache_position explicitly.
        # HF's _get_initial_cache_position computes:
        #   cache_position = [0, ..., seq_len-1][past_length:]
        # When seq_len=1 (our single token) and past_length=N (KV cache),
        # this gives an EMPTY tensor → IndexError on cache_position[-1].
        # By passing cache_position ourselves, HF skips that computation.
        # The first token's position is self.current_cache_len (right after
        # all cached tokens).
        cache_position = torch.tensor(
            [self.current_cache_len], dtype=torch.long, device=self.device
        )

        with torch.no_grad():
            gen_outputs = self.model.generate(
                input_ids=first_token_id,
                attention_mask=attention_mask,
                past_key_values=self.past_key_values,
                cache_position=cache_position,
                max_new_tokens=max_new_tokens - 1,  # -1 because first token is already selected
                do_sample=False,
                temperature=None,
                top_p=None,
                return_dict_in_generate=True,
                use_cache=True,
            )

        # gen_outputs.sequences = [first_token_id, generated_token_1, ...]
        # The first token in sequences is our input (first_token_id)
        all_generated = gen_outputs.sequences[0].tolist()  # includes first_token_id

        # Update past_key_values from generation output
        if hasattr(gen_outputs, 'past_key_values') and gen_outputs.past_key_values is not None:
            self.past_key_values = gen_outputs.past_key_values

        # Update cache length: add all generated tokens (including first_token_id)
        self.current_cache_len += len(all_generated)

        self.timing_stats["decode_time"] += time.time() - t0

        # Track generated ids for future repetition penalty
        self._all_token_ids.extend(all_generated)

        # Update manager cache len
        if self.kv_manager:
            self.kv_manager.current_cache_len = self.current_cache_len

        response_text = self.tokenizer.decode(all_generated, skip_special_tokens=True)
        return response_text.strip(), len(all_generated)

    def get_cache_len(self):
        """Get current KV cache length."""
        return self.current_cache_len

    def get_stats(self):
        """Get combined timing and pruning stats."""
        stats = dict(self.timing_stats)
        if self.kv_manager:
            stats.update(self.kv_manager.get_stats())
        stats["current_cache_len"] = self.current_cache_len
        return stats
