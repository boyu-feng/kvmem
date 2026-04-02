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

        Uses model.generate() directly for optimal performance (matching baseline).
        Extracts past_key_values from the generation output for subsequent
        incremental steps.

        After generation, if the response contains content beyond the first
        Action line (e.g., hallucinated future Observations), we truncate
        both the response text AND the KV cache to keep only the useful portion.

        Args:
            prompt_text: the full initial prompt (system + question + trajectory prefix)
            max_new_tokens: maximum tokens to generate
            stop_strings: list of strings that should trigger truncation.
                         If the generated text contains any of these strings,
                         the text and KV cache are truncated at that point.

        Returns:
            response_text: generated text (truncated to useful portion)
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

        # Use model.generate() directly — this handles both prefill and decode
        # in HuggingFace's optimized pipeline, matching baseline performance.
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

        # Extract generated token IDs (exclude prompt)
        generated_ids = gen_outputs.sequences[0][prompt_len:].tolist()

        # Extract past_key_values for subsequent incremental steps
        if hasattr(gen_outputs, 'past_key_values') and gen_outputs.past_key_values is not None:
            self.past_key_values = gen_outputs.past_key_values
        else:
            # Fallback: run a forward pass to get past_key_values
            # This should not happen with return_dict_in_generate=True
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

        full_pkv = self.past_key_values  # Tuple of (key, value) pairs for each layer

        prompt_kv = []
        generated_kv = []

        for layer_idx in range(len(full_pkv)):
            # 每层是一个元组 (key_states, value_states)
            k, v = full_pkv[layer_idx]
            
            # 维度通常是 [batch_size, num_heads, seq_len, head_dim]
            # 我们对 seq_len 维度进行切片
            
            # 1. 提取 Prompt 部分 (从 0 到 prompt_len)
            pk = k[:, :, :prompt_len, :]
            pv = v[:, :, :prompt_len, :]
            prompt_kv.append((pk, pv))
            
            # 2. 提取新生成的 Thought/Action 部分 (从 prompt_len 到最后)
            gk = k[:, :, prompt_len:, :]
            gv = v[:, :, prompt_len:, :]
            generated_kv.append((gk, gv))

        # 转换为元组结构，保持与 Transformers 格式一致
        prompt_kv = tuple(prompt_kv)
        generated_kv = tuple(generated_kv)

        # Approximate timing split (generate handles both prefill and decode)
        self.timing_stats["prefill_time"] += total_time * 0.3  # rough estimate
        self.timing_stats["decode_time"] += total_time * 0.7

        # Register initial cache with manager (use prompt_len as the protected prefix)
        if self.kv_manager:
            self.kv_manager.register_initial_cache(prompt_len)
            self.kv_manager.current_cache_len = self.current_cache_len

        # Track all token ids for repetition penalty
        self._all_token_ids = input_ids[0].tolist() + generated_ids

        response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Truncate at stop_strings if found (remove hallucinated future content)
        if stop_strings:
            truncated_text = response_text
            for stop_str in stop_strings:
                idx = truncated_text.find(stop_str)
                if idx != -1:
                    truncated_text = truncated_text[:idx]
            
            if len(truncated_text) < len(response_text):
                # Re-tokenize the truncated text to find how many tokens to keep
                truncated_ids = self.tokenizer(
                    truncated_text, add_special_tokens=False
                ).input_ids
                keep_count = prompt_len + len(truncated_ids)
                self.truncate_cache(keep_count)
                response_text = truncated_text
        
        return response_text.strip(), prompt_kv, generated_kv

    def generate_incremental(self, new_text, max_new_tokens=256, stop_strings=None):
        """
        Subsequent calls: only encode new_text, reuse cached KV.

        IMPORTANT: To match baseline behavior, we do NOT wrap new_text with role markers.
        In baseline, the entire trajectory (Thought/Action/Observation sequence) is inside
        a single user message, and the model generates continuously. Here, we:
        1. Keep all content (initial prompt + model output + observation) in the same
           KV cache without role switching
        2. Append observation directly, allowing the model to continue as if it were
           continuing from within the same user message context

        The KV cache ends at the model's generated tokens (Thought/Action).
        We append Observation directly, and the model generates the next Thought/Action.
        This matches baseline's behavior where observation is part of the running trajectory.

        After generation, the response is truncated at stop_strings to prevent
        hallucinated future content from polluting the KV cache.

        Args:
            new_text: new text to append (e.g., Observation content)
            max_new_tokens: maximum tokens to generate
            stop_strings: list of strings that should trigger truncation

        Returns:
            response_text: generated text (truncated to useful portion)
        """
        assert self.past_key_values is not None, \
            "Must call generate_first() before generate_incremental()"

        # CRITICAL FIX: Do NOT wrap with role markers to match baseline behavior.
        # In baseline, the trajectory (Thought/Action/Observation) is all in one user message.
        # The model sees Observation as continuation of the reasoning trace, NOT a new turn.
        # Simply append the new_text (Observation) directly.
        # This ensures the model's attention pattern matches the baseline.
        wrapped_text = new_text

        # Tokenize only the wrapped text
        new_input_ids = self.tokenizer(
            wrapped_text, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(self.device)
        new_token_count = new_input_ids.shape[1]

        # Prefill new tokens with existing KV cache
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
        self.past_key_values = outputs.past_key_values
        self.current_cache_len += new_token_count
        prefill_time = time.time() - t0
        self.timing_stats["prefill_time"] += prefill_time

        step_kv = []
        # outputs.past_key_values 结构为: ( (layer_0_k, layer_0_v), (layer_1_k, layer_1_v), ... )
        for layer_pkv in outputs.past_key_values:
            k, v = layer_pkv  # 形状通常为 [batch_size, num_heads, seq_len, head_dim]
            
            # 提取最后 new_token_count 个 token 对应的 KV 矩阵
            # 使用 .detach() 断开计算图，避免内存泄露
            # 使用 .clone() 确保即使原始 cache 被修改或 prune，这部分数据依然完整
            s_k = k[:, :, -new_token_count:, :].detach().clone()
            s_v = v[:, :, -new_token_count:, :].detach().clone()
            
            step_kv.append((s_k, s_v))

        # 转换为元组，方便后续传给你的 Delta Rule 算法
        step_kv = tuple(step_kv)

        # Update manager
        if self.kv_manager:
            self.kv_manager.append_step(new_token_count)
            self.kv_manager.current_cache_len = self.current_cache_len

        # Check if pruning is needed
        if self.kv_manager and self.kv_manager.should_prune():
            piggyback_attentions = outputs.attentions if need_attention else None
            self._do_pruning(piggyback_attentions, step_kv=step_kv, step_token_count=new_token_count)

        # Track new token ids for repetition penalty
        self._all_token_ids.extend(new_input_ids[0].tolist())

        # Record cache length before decode, needed for truncation
        cache_len_before_decode = self.current_cache_len

        # Decode using model.generate() for optimized autoregressive generation.
        # We pass the last logits to get the first decoded token, then let
        # model.generate() handle the rest efficiently.
        response_text, generated_len = self._decode(
            outputs.logits, max_new_tokens
        )

        # Truncate at stop_strings if found (remove hallucinated future content)
        if stop_strings and response_text:
            truncated_text = response_text
            for stop_str in stop_strings:
                idx = truncated_text.find(stop_str)
                if idx != -1:
                    truncated_text = truncated_text[:idx]
            
            if len(truncated_text) < len(response_text):
                # Re-tokenize the truncated text to find how many tokens to keep
                truncated_ids = self.tokenizer(
                    truncated_text, add_special_tokens=False
                ).input_ids
                keep_count = cache_len_before_decode + len(truncated_ids)
                self.truncate_cache(keep_count)
                response_text = truncated_text

        return response_text.strip() if response_text else response_text

    def _split_and_save_triad_kv(self, obs_len, full_response_text):
        """
        将 KV Cache 拆分为: 1. Observation, 2. Thought, 3. Action
        """
        # 1. 文本层面定位 Action 的起始位置
        # 假设格式包含 "Action:" 关键字
        action_start_idx = full_response_text.find("Action:")
        
        if action_start_idx != -1:
            thought_text = full_response_text[:action_start_idx]
            action_text = full_response_text[action_start_idx:]
        else:
            # 如果没找到 Action，则全部归为 Thought
            thought_text = full_response_text
            action_text = ""

        # 2. 将文本转换为 Token 长度以便切分 KV 矩阵
        # 注意：这里需要精确匹配，不加 special tokens
        thought_tokens = self.tokenizer(thought_text, add_special_tokens=False).input_ids
        action_tokens = self.tokenizer(action_text, add_special_tokens=False).input_ids
        
        thought_len = len(thought_tokens)
        action_len = len(action_tokens)

        # 3. 计算在当前全量 KV 中的索引位置
        # 当前总长度 = 历史 + obs_len + thought_len + action_len
        end_idx = self.current_cache_len
        action_start = end_idx - action_len
        thought_start = action_start - thought_len
        obs_start = thought_start - obs_len

        step_storage = {"observation": [], "thought": [], "action": []}

        for layer_idx in range(len(self.past_key_values)):
            k, v = self.past_key_values[layer_idx]
            
            # 辅助切片函数 (移至 CPU 避免显存爆炸)
            def get_slice(s, e): 
                return (k[:, :, s:e, :].detach().cpu(), 
                        v[:, :, s:e, :].detach().cpu())

            step_storage["observation"].append(get_slice(obs_start, thought_start))
            step_storage["thought"].append(get_slice(thought_start, action_start))
            step_storage["action"].append(get_slice(action_start, end_idx))

        # 保存到类属性中
        if not hasattr(self, 'triad_kv_history'):
            self.triad_kv_history = []
        
        self.triad_kv_history.append(step_storage)
        
        print(f"KV Split Done - Obs: {obs_len}, Thought: {thought_len}, Action: {action_len}")

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
