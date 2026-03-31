import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import json
import os
from datetime import datetime
# ==================== LLM Wrapper ====================
class QwenLLM:
    """Wrapper for local Qwen2.5-7B-Instruct model."""

    def __init__(self, model_path):
        print(f"[INFO] Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        print("[INFO] Model loaded successfully.")

    def _get_past_kv_seq_len(self, past_key_values):
        """从 past_key_values 中推断缓存的 token 长度（兼容不同 shape 布局）。"""
        if past_key_values is None:
            return 0
        seq_lens = []
        for layer in past_key_values:
            # layer 可能是 (k, v) tuple
            k = layer[0] if isinstance(layer, (list, tuple)) else layer
            if k is None:
                continue
            s = k.shape  # 常见形状： (batch, num_heads, seq_len, head_dim) 或 (batch, seq_len, num_heads, head_dim)
            if len(s) >= 4:
                # 取中间较小维度作为 seq_len 的可靠近似；用 max 作为保守估计
                seq_len = max(s[1], s[2])
            elif len(s) >= 2:
                seq_len = s[-2]
            else:
                seq_len = s[0]
            seq_lens.append(int(seq_len))
        # 多层应一致，取第一个或最大以保险
        return int(seq_lens[0]) if seq_lens else 0

    def _save_state_as_json(self, state: dict, path: str):
        """
        将可序列化的 state 字典写入 JSON 文件（自动创建目录）。
        注意：state 应该已被转换为基础 Python 类型（list/int/float/str）。
        """
        if not path:
            raise ValueError("save path must be provided")
        dirname = os.path.dirname(path) or "."
        os.makedirs(dirname, exist_ok=True)
        # 使用 ensure_ascii=False 保持 unicode 可读，indent 便于查看
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def generate(self, prompt, max_new_tokens=256, save_state_path: str = None):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        device = self.model.device
        is_cuda = getattr(device, "type", "") == "cuda"

        # 采集前置统计
        prefill_time = None
        gen_time = None
        kv_len = None
        mem_before = mem_after = mem_peak = None

        # 先做一次前向以获得 past_key_values（不改动模型状态，仅用于观测）
        with torch.no_grad():
            if is_cuda:
                torch.cuda.reset_peak_memory_stats(device)
                mem_before = torch.cuda.memory_allocated(device)
            t0 = time.time()
            outputs_prefill = self.model(**inputs, use_cache=True)
            prefill_time = time.time() - t0
            past = getattr(outputs_prefill, "past_key_values", None)
            kv_len = self._get_past_kv_seq_len(past)

            # 记录显存情况
            if is_cuda:
                mem_after = torch.cuda.memory_allocated(device)
                mem_peak = torch.cuda.max_memory_allocated(device)

            # 真实生成（使用 HF generate；注意会重复做一次编码）
            t0 = time.time()
            gen_outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                return_dict_in_generate=True,
            )
            gen_time = time.time() - t0

        generated_ids = gen_outputs.sequences[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # 打印/返回统计（可按需调整为 logging 或返回值）
        # 构建可序列化状态：把 tensor 转为 list，把可能的 numpy/torch 值转为基础类型
        def _to_py(x):
            if isinstance(x, torch.Tensor):
                try:
                    return x.cpu().tolist()
                except Exception:
                    return None
            # 对于 numpy types 或其他可直接序列化的类型，尝试直接转
            try:
                if hasattr(x, "item"):
                    return x.item()
            except Exception:
                pass
            return x

        generated_ids_list = _to_py(generated_ids)
        input_ids_list = _to_py(inputs.get("input_ids"))

        stats = {
            "kv_cache_len": kv_len,
            "prefill_time_s": prefill_time,
            "generate_time_s": gen_time,
            "cuda_mem_before_bytes": mem_before,
            "cuda_mem_after_bytes": mem_after,
            "cuda_mem_peak_bytes": mem_peak,
            "prompt": prompt,
            "input_ids": input_ids_list,
            "generated_ids": generated_ids_list,
            "response": response.strip(),
            "device": str(device),
            "saved_at": datetime.utcnow().isoformat() + "Z",
        }

        # 如果指定了保存路径，则写入 JSON
        if save_state_path:
            try:
                self._save_state_as_json(stats, save_state_path)
                print(f"[STATS SAVED] {save_state_path}")
            except Exception as e:
                print(f"[ERROR] saving stats to {save_state_path}: {e}")

        # 打印统计（仍然保持原行为）
        print(f"[STATS] {stats}")

        return response.strip()

