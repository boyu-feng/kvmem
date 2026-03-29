from typing import Iterable, Tuple, Optional, List, Any, Dict
import torch

class OurCompressor:
	"""
	纯粹融合器：仅做 fusion 汽作（不负责索引 / device / DynamicCache 构建）。
	调用约定：
	  - base_layers: List[ (base_k, base_v, suffix_k, suffix_v) ]（由 pruning_strategy 生成）
	  - step_kv: Optional[List[(k_step, v_step)]] （每层一步的 kv）
	  - step_token_count: int
	返回：
	  - final_layers: List[(k_merged, v_merged)]
	  - added_tokens: int
	  - used_step: bool
	  - note: str
	"""

	def __init__(self):
		pass

	def merge(self,
			  base_layers: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
			  step_kv: Optional[Iterable[Tuple[torch.Tensor, torch.Tensor]]] = None,
			  step_token_count: int = 0) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], int, bool, str]:
		"""
		把 step_kv 插入到 base（base = prefix + pooled）与 suffix 之间：
		    final = [base] + [step_kv] + [suffix]
		如果 step_kv 缺失或合并失败，回退为 base + suffix。
		"""
		final_layers: List[Tuple[torch.Tensor, torch.Tensor]] = []
		added_tokens = 0
		used_step = False
		note = "no_step_kv"

		# 快速检查 step_kv 的合法性（如果提供）
		has_step = step_kv is not None and step_token_count and hasattr(step_kv, "__len__") and len(step_kv) == len(base_layers)

		if has_step:
			try:
				for layer_idx, (base_k, base_v, suffix_k, suffix_v) in enumerate(base_layers):
					k_step, v_step = step_kv[layer_idx]
					# 合并顺序： base + step + suffix
					k_merged = torch.cat([base_k, k_step, suffix_k], dim=2)
					v_merged = torch.cat([base_v, v_step, suffix_v], dim=2)
					final_layers.append((k_merged, v_merged))
				added_tokens = int(step_token_count)
				used_step = True
				note = "merged_with_step_kv"
			except Exception as e:
				# 合并失败则回退为 base + suffix
				final_layers = []
				for base_k, base_v, suffix_k, suffix_v in base_layers:
					k_merged = torch.cat([base_k, suffix_k], dim=2)
					v_merged = torch.cat([base_v, suffix_v], dim=2)
					final_layers.append((k_merged, v_merged))
				added_tokens = 0
				used_step = False
				note = f"step_kv_merge_failed: {e}"
		else:
			# 未提供 step_kv：直接拼回 suffix
			for base_k, base_v, suffix_k, suffix_v in base_layers:
				k_merged = torch.cat([base_k, suffix_k], dim=2)
				v_merged = torch.cat([base_v, suffix_v], dim=2)
				final_layers.append((k_merged, v_merged))
			note = "no_step_kv"

		return final_layers, added_tokens, used_step, note