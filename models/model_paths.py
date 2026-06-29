"""Resolve local model directories for offline inference."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

DEFAULT_QWEN_REPO = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_LLAMA_REPO = "meta-llama/Meta-Llama-3.1-8B-Instruct"

MODEL_FAMILIES: Dict[str, Dict[str, str]] = {
    "qwen": {
        "repo_id": DEFAULT_QWEN_REPO,
        "dir_name": "Qwen2.5-7B-Instruct",
        "label": "Qwen2.5-7B-Instruct",
    },
    "llama": {
        "repo_id": DEFAULT_LLAMA_REPO,
        "dir_name": "Meta-Llama-3.1-8B-Instruct",
        "label": "Llama-3.1-8B-Instruct",
    },
}


class AmbiguousModelError(FileNotFoundError):
    """Raised when multiple supported models are available under auto mode."""


def is_local_model_dir(path: str) -> bool:
    return bool(path) and os.path.isdir(path) and os.path.isfile(os.path.join(path, "config.json"))


def _hf_repo_basename(repo_id: str) -> str:
    return repo_id.strip().split("/")[-1]


def _models_roots() -> List[str]:
    hf_home = os.environ.get("HF_HOME", "/root/autodl-tmp/hf_cache").strip()
    roots: List[str] = []
    if hf_home:
        roots.append(os.path.join(hf_home, "models"))
    roots.append("/root/autodl-tmp/hf_cache/models")
    out: List[str] = []
    seen = set()
    for root in roots:
        if root and root not in seen:
            seen.add(root)
            out.append(root)
    return out


def infer_model_family(path: str) -> Optional[str]:
    lower = (path or "").lower()
    if "qwen" in lower:
        return "qwen"
    if "llama" in lower:
        return "llama"

    config_path = os.path.join(path, "config.json")
    if not os.path.isfile(config_path):
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    blob = " ".join(
        str(cfg.get(key, ""))
        for key in ("model_type", "architectures", "_name_or_path")
    ).lower()
    if "qwen" in blob:
        return "qwen"
    if "llama" in blob:
        return "llama"
    return None


def find_local_model_for_family(family: str) -> Optional[str]:
    spec = MODEL_FAMILIES.get(family)
    if spec is None:
        return None
    for root in _models_roots():
        cand = os.path.join(root, spec["dir_name"])
        if is_local_model_dir(cand):
            return cand
    return None


def detect_available_models() -> Dict[str, str]:
    """Return {family: local_path} for each supported model found locally."""
    found: Dict[str, str] = {}
    for family in MODEL_FAMILIES:
        path = find_local_model_for_family(family)
        if path:
            found[family] = path
    return found


def local_model_candidates() -> List[str]:
    candidates = [
        os.environ.get("KVMEM_MODEL_PATH", "").strip(),
        os.environ.get("LOCAL_MODEL_PATH", "").strip(),
    ]
    for family in MODEL_FAMILIES:
        path = find_local_model_for_family(family)
        if path:
            candidates.append(path)
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
    for root in _models_roots():
        cand = os.path.join(root, basename)
        if is_local_model_dir(cand):
            return cand
    return None


def default_local_model_dir(repo_id: str = DEFAULT_QWEN_REPO) -> str:
    basename = _hf_repo_basename(repo_id)
    hf_home = os.environ.get("HF_HOME", "/root/autodl-tmp/hf_cache").strip()
    return os.path.join(hf_home, "models", basename)


def download_hf_model(repo_id: str, local_dir: str) -> None:
    """Download a HuggingFace model snapshot into a local directory."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise FileNotFoundError(
            "huggingface_hub is required to download models. "
            "Install it or pass --model_path to an existing local model directory."
        ) from exc

    os.makedirs(local_dir, exist_ok=True)
    print(f"[INFO] Downloading {repo_id} -> {local_dir}")
    snapshot_download(repo_id=repo_id, local_dir=local_dir)


def _resolve_auto_model_path(model_family: str = "auto") -> str:
    model_family = (model_family or "auto").strip().lower()
    if model_family not in ("auto", *MODEL_FAMILIES):
        raise ValueError(f"Unsupported model_family: {model_family!r}")

    for env_name in ("KVMEM_MODEL_PATH", "LOCAL_MODEL_PATH"):
        env_path = os.environ.get(env_name, "").strip()
        if is_local_model_dir(env_path):
            family = infer_model_family(env_path) or "custom"
            print(f"[INFO] Using model from {env_name}: {env_path} ({family})")
            return env_path

    available = detect_available_models()
    if model_family in MODEL_FAMILIES:
        path = available.get(model_family)
        if path:
            print(
                f"[INFO] Auto-selected {MODEL_FAMILIES[model_family]['label']} "
                f"via --model_family {model_family}: {path}"
            )
            return path
        raise FileNotFoundError(
            f"Local {MODEL_FAMILIES[model_family]['label']} not found. "
            f"Searched under: {', '.join(_models_roots())}"
        )

    if not available:
        raise FileNotFoundError(
            "No local Qwen or Llama model found. Pass --model_path, set "
            "KVMEM_MODEL_PATH / LOCAL_MODEL_PATH, or use --model_family qwen|llama "
            "to download one."
        )
    if len(available) == 1:
        family, path = next(iter(available.items()))
        print(
            f"[INFO] Auto-detected {MODEL_FAMILIES[family]['label']} "
            f"({family}): {path}"
        )
        return path

    lines = [
        "Both Qwen and Llama models are available locally; auto mode cannot choose.",
        "Pass --model_path <dir> or --model_family qwen|llama.",
        "Found:",
    ]
    for family, path in sorted(available.items()):
        lines.append(f"  - {family}: {path}")
    raise AmbiguousModelError("\n".join(lines))


def resolve_local_model_path(explicit: str = "auto", *, model_family: str = "auto") -> str:
    """
    Pick a local model directory for analysis/inference.

    - explicit local path -> use directly
    - HF repo id -> map to $HF_HOME/models/<basename>
    - "auto" -> detect Qwen/Llama; error if both exist unless model_family is set
    """
    explicit = (explicit or "auto").strip()
    if is_local_model_dir(explicit):
        return explicit

    known_repos = {spec["repo_id"] for spec in MODEL_FAMILIES.values()}
    if explicit not in ("auto", *known_repos):
        mapped = _local_path_for_hf_repo(explicit)
        if mapped:
            return mapped
        if os.path.isdir(explicit):
            raise FileNotFoundError(
                f"Model path exists but is not a valid local model dir (missing config.json): {explicit}"
            )

    return _resolve_auto_model_path(model_family=model_family)


def ensure_local_model_path(
    explicit: str = "auto",
    *,
    model_family: str = "auto",
    allow_download: bool = True,
) -> str:
    """
    Resolve a local model directory, optionally downloading when none exists.

    When no local model is found and allow_download is True, pass
    --model_family qwen or --model_family llama to choose which one to download.
    """
    explicit = (explicit or "auto").strip()
    model_family = (model_family or "auto").strip().lower()

    try:
        return resolve_local_model_path(explicit, model_family=model_family)
    except FileNotFoundError:
        if not allow_download:
            raise

    if explicit not in ("auto", *{spec["repo_id"] for spec in MODEL_FAMILIES.values()}):
        raise

    if model_family not in MODEL_FAMILIES:
        raise FileNotFoundError(
            "No local model found and model_family is auto. "
            "Pass --model_family qwen or --model_family llama to download one."
        )

    repo_id = MODEL_FAMILIES[model_family]["repo_id"]
    local_dir = default_local_model_dir(repo_id)
    if is_local_model_dir(local_dir):
        return local_dir

    download_hf_model(repo_id, local_dir)
    if is_local_model_dir(local_dir):
        print(
            f"[INFO] Using downloaded {MODEL_FAMILIES[model_family]['label']}: "
            f"{local_dir}"
        )
        return local_dir

    raise FileNotFoundError(
        f"Download finished but model dir is still invalid (missing config.json): {local_dir}"
    )
