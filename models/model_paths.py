"""Resolve local model directories for offline inference."""

from __future__ import annotations

import os
from typing import List

DEFAULT_QWEN_REPO = "Qwen/Qwen2.5-7B-Instruct"


def is_local_model_dir(path: str) -> bool:
    return bool(path) and os.path.isdir(path) and os.path.isfile(os.path.join(path, "config.json"))


def local_qwen_model_candidates() -> List[str]:
    hf_home = os.environ.get("HF_HOME", "").strip()
    candidates = [
        os.environ.get("KVMEM_MODEL_PATH", "").strip(),
        os.environ.get("LOCAL_MODEL_PATH", "").strip(),
        "/root/autodl-tmp/hf_cache/models/Qwen2.5-7B-Instruct",
        os.path.join(hf_home, "models", "Qwen2.5-7B-Instruct") if hf_home else "",
    ]
    out: List[str] = []
    seen = set()
    for cand in candidates:
        if not cand or cand in seen:
            continue
        seen.add(cand)
        out.append(cand)
    return out


def resolve_local_model_path(explicit: str = "auto") -> str:
    """
    Pick a local model directory for analysis/inference.

    - explicit local path -> use directly
    - "auto" or HF repo id -> search known local Qwen paths only
    """
    explicit = (explicit or "auto").strip()
    if is_local_model_dir(explicit):
        return explicit

    if explicit not in ("auto", DEFAULT_QWEN_REPO) and os.path.isdir(explicit):
        raise FileNotFoundError(
            f"Model path exists but is not a valid local model dir (missing config.json): {explicit}"
        )

    for cand in local_qwen_model_candidates():
        if is_local_model_dir(cand):
            return cand

    searched = ", ".join(local_qwen_model_candidates()) or "(none)"
    raise FileNotFoundError(
        "Local Qwen model not found. Set --model_path to your local model directory, "
        f"or export KVMEM_MODEL_PATH. Searched: {searched}"
    )
