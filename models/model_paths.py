"""Resolve local model directories for offline inference."""

from __future__ import annotations

import os
from typing import List, Optional

DEFAULT_QWEN_REPO = "Qwen/Qwen2.5-7B-Instruct"

# Default local model folder under $HF_HOME/models.
LOCAL_MODEL_DIR_NAMES = [
    "Qwen2.5-7B-Instruct",
]


def is_local_model_dir(path: str) -> bool:
    return bool(path) and os.path.isdir(path) and os.path.isfile(os.path.join(path, "config.json"))


def _hf_repo_basename(repo_id: str) -> str:
    return repo_id.strip().split("/")[-1]


def local_model_candidates() -> List[str]:
    hf_home = os.environ.get("HF_HOME", "/root/autodl-tmp/hf_cache").strip()
    models_root = os.path.join(hf_home, "models") if hf_home else ""
    candidates = [
        os.environ.get("KVMEM_MODEL_PATH", "").strip(),
        os.environ.get("LOCAL_MODEL_PATH", "").strip(),
    ]
    for name in LOCAL_MODEL_DIR_NAMES:
        candidates.append(os.path.join(models_root, name))
        candidates.append(os.path.join("/root/autodl-tmp/hf_cache/models", name))
    out: List[str] = []
    seen = set()
    for cand in candidates:
        if not cand or cand in seen:
            continue
        seen.add(cand)
        out.append(cand)
    return out


def local_qwen_model_candidates() -> List[str]:
    return local_model_candidates()


def _local_path_for_hf_repo(repo_id: str) -> Optional[str]:
    if not repo_id or "/" not in repo_id:
        return None
    basename = _hf_repo_basename(repo_id)
    hf_home = os.environ.get("HF_HOME", "/root/autodl-tmp/hf_cache").strip()
    for root in (os.path.join(hf_home, "models"), "/root/autodl-tmp/hf_cache/models"):
        if not root:
            continue
        cand = os.path.join(root, basename)
        if is_local_model_dir(cand):
            return cand
    return None


def resolve_local_model_path(explicit: str = "auto") -> str:
    """
    Pick a local model directory for analysis/inference.

    - explicit local path -> use directly
    - HF repo id (e.g. Qwen/Qwen2.5-7B-Instruct) -> map to $HF_HOME/models/<basename>
    - "auto" -> search known local model paths (no HuggingFace download)
    """
    explicit = (explicit or "auto").strip()
    if is_local_model_dir(explicit):
        return explicit

    if explicit not in ("auto", DEFAULT_QWEN_REPO):
        mapped = _local_path_for_hf_repo(explicit)
        if mapped:
            return mapped
        if os.path.isdir(explicit):
            raise FileNotFoundError(
                f"Model path exists but is not a valid local model dir (missing config.json): {explicit}"
            )

    for cand in local_model_candidates():
        if is_local_model_dir(cand):
            return cand

    searched = ", ".join(local_model_candidates()) or "(none)"
    raise FileNotFoundError(
        "Local model not found. Set --model_path to your local model directory, "
        f"or export KVMEM_MODEL_PATH / LOCAL_MODEL_PATH. Searched: {searched}"
    )
