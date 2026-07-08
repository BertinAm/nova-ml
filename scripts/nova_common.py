"""Shared utilities for the NOVA ML pipeline.

Works identically on Kaggle, Colab, or a local machine:

- HF_USERNAME / repo naming: everything publishes under the personal
  account ``unixio`` (override with the NOVA_HF_USERNAME env var).
- HF_TOKEN resolution order: explicit argument > Kaggle secret named
  "HF_TOKEN" > HF_TOKEN env var > cached huggingface-cli login.
- Environment detection so scripts can pick sensible paths
  (/kaggle/working on Kaggle, ./ locally).
"""
import hashlib
import os
from pathlib import Path

HF_USERNAME = os.environ.get("NOVA_HF_USERNAME", "unixio")

# HF repo per module — personal-account equivalents of the doc's
# nova-assistive org repos.
HF_REPOS = {
    "MOD-01": f"{HF_USERNAME}/nova-obstacle-detection",
    "MOD-04": f"{HF_USERNAME}/nova-currency-detection",
    "MOD-05-detect": f"{HF_USERNAME}/nova-face-detection",
    "MOD-05-embed": f"{HF_USERNAME}/nova-face-embedding",
}


def is_kaggle() -> bool:
    return os.path.exists("/kaggle/working")


def work_dir() -> Path:
    """Writable output root: /kaggle/working on Kaggle, cwd elsewhere."""
    return Path("/kaggle/working") if is_kaggle() else Path(".")


def dataset_dir() -> Path:
    """Where input datasets live: /kaggle/input on Kaggle (attached
    datasets, read-only), ./datasets elsewhere."""
    return Path("/kaggle/input") if is_kaggle() else Path("datasets")


def get_hf_token(explicit: str | None = None) -> str:
    """Resolve the HuggingFace token without ever hardcoding it."""
    if explicit:
        return explicit

    if is_kaggle():
        try:
            from kaggle_secrets import UserSecretsClient

            return UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            pass  # fall through to env var

    token = os.environ.get("HF_TOKEN")
    if token:
        return token

    # Last resort: whatever `huggingface-cli login` cached.
    from huggingface_hub import HfFolder

    cached = HfFolder.get_token()
    if cached:
        return cached
    raise RuntimeError(
        "No HuggingFace token found. On Kaggle: Add-ons > Secrets > add 'HF_TOKEN'. "
        "Locally: export HF_TOKEN=... or run `huggingface-cli login`."
    )


def sha256_of(file_path: str | Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dirs() -> dict[str, Path]:
    """Create the standard output directories and return them."""
    root = work_dir()
    dirs = {
        "checkpoints": root / "checkpoints",
        "exports": root / "exports",
        "evaluation": root / "evaluation",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs
