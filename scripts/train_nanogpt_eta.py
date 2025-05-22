#!/usr/bin/env python3
"""Wrapper script to train *nanoGPT* with Dedekind–η activations.

Usage
-----
Examples for single-node, 8×H100 training (run from project root)::

    scripts/train_nanogpt_eta.py --clone-if-missing --repo-dir /scratch/$USER/nanogpt \
        -- --batch_size 12 --gradient_accumulation_steps 40

To launch distributed training with *torchrun* (PyTorch >= 2.0)::

    torchrun --standalone --nproc_per_node=8 scripts/train_nanogpt_eta.py \
        --clone-if-missing -- --batch_size 12

All arguments after the ``--`` separator are forwarded verbatim to
``nanoGPT/train.py``.

The script will:
1. Clone the `nanoGPT` repository if necessary.
2. Inject our :class:`modular_form_activation.activations.DedekindEtaActivation`
   into the model definition.
3. Execute the upstream ``train.py`` in-process so that distributed
   initialisation (``torchrun``) picks up the same Python interpreter.

Logging
~~~~~~~
If the environment variable ``WANDB_API_KEY`` is defined, the script will
configure *Weights & Biases* logging automatically and set sensible defaults
(W&B project: ``eta_nanogpt``, run name derived from the UTC timestamp). This
should satisfy the requirements of Keller Jordan's competition scoreboard.
"""

from __future__ import annotations

import argparse
import os
import runpy
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

REPO_URL = "https://github.com/karpathy/nanoGPT.git"
DEFAULT_REPO_DIR = Path("extern/nanogpt")


def _clone_repo(repo_dir: Path) -> None:
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"[eta-nanogpt] Cloning {REPO_URL} into {repo_dir} …", flush=True)
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(repo_dir)], check=True)


def _maybe_configure_wandb() -> None:  # noqa: D401 – side-effect function
    """Set default W&B config if the API key is present."""

    if "WANDB_API_KEY" not in os.environ:
        return  # user disabled W&B

    os.environ.setdefault("wandb_log", "True")
    os.environ.setdefault("WANDB_PROJECT", "eta_nanogpt")
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    os.environ.setdefault("WANDB_RUN_NAME", f"eta_{timestamp}")


def parse_cli(argv: List[str] | None = None) -> tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=DEFAULT_REPO_DIR,
        help="Local path of the nanoGPT checkout (will be placed on PYTHONPATH).",
    )
    parser.add_argument(
        "--clone-if-missing",
        action="store_true",
        help="Clone the repository if the path does not exist.",
    )
    parser.add_argument(
        "--", dest="train_argv", nargs=argparse.REMAINDER, help="Args forwarded to nanoGPT/train.py."
    )
    args = parser.parse_args(argv)
    train_argv = args.train_argv[1:] if args.train_argv and args.train_argv[0] == "--" else args.train_argv
    train_argv = train_argv or []
    return args, train_argv


def main(argv: List[str] | None = None) -> None:  # noqa: D401 – CLI entry point
    args, train_argv = parse_cli(argv)

    repo_dir: Path = args.repo_dir
    if not repo_dir.exists():
        if args.clone_if_missing:
            _clone_repo(repo_dir)
        else:
            raise FileNotFoundError(
                f"nanoGPT repo directory {repo_dir} does not exist. Use --clone-if-missing to clone automatically."
            )

    # Put nanoGPT on the import path **before** we patch it.
    sys.path.insert(0, str(repo_dir))

    # Configure logging defaults.
    _maybe_configure_wandb()

    # Monkey-patch the MLP.
    from modular_form_activation.integrations.nanogpt_patch import patch_nanogpt

    patch_nanogpt()

    # Build argv for train.py and execute it in-process.
    sys.argv = [str(repo_dir / "train.py")] + train_argv
    print(f"[eta-nanogpt] Launching nanoGPT/train.py with argv: {sys.argv[1:]}")
    runpy.run_path(repo_dir / "train.py", run_name="__main__")


if __name__ == "__main__":  # pragma: no cover – script entry
    main() 