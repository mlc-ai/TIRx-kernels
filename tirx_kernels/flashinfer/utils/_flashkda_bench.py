# Copyright (c) 2026 The TIRx Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""MoonshotAI/FlashKDA benchmark adapter for the recurrent-KDA port."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import torch


@dataclass
class FlashKDARawReference:
    launch: Callable[[], None]
    provenance: dict[str, Any]
    correctness: dict[str, float]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True, stderr=subprocess.STDOUT
        ).strip()
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"failed to verify FlashKDA source provenance at {root}: {error.output.strip()}"
        ) from error


def _source_dir(dist: Any) -> Path | None:
    override = os.environ.get("FLASHKDA_SOURCE_DIR")
    if override:
        return Path(override).expanduser().resolve(strict=True)

    direct_url_text = dist.read_text("direct_url.json")
    if direct_url_text:
        url = json.loads(direct_url_text).get("url", "")
        parsed = urlparse(url)
        if parsed.scheme == "file":
            source_dir = Path(unquote(parsed.path))
            if source_dir.exists():
                return source_dir.resolve(strict=True)
    return None


def _load_flash_kda_peer() -> tuple[Any, dict[str, Any]]:
    try:
        dist = distribution("flash-kda")
    except PackageNotFoundError as error:
        raise RuntimeError("MoonshotAI/FlashKDA is not installed") from error

    flash_kda = import_module("flash_kda")
    extension = import_module("flash_kda_C")
    package_path = Path(flash_kda.__file__).resolve(strict=True)
    extension_path = Path(extension.__file__).resolve(strict=True)
    provenance = {
        "repository": "https://github.com/MoonshotAI/FlashKDA.git",
        "package_version": dist.version,
        "package_path": str(package_path),
        "package_sha256": _sha256(package_path),
        "extension_path": str(extension_path),
        "extension_sha256": _sha256(extension_path),
    }

    source_dir = _source_dir(dist)
    if source_dir is not None:
        source_commit = _git_output(source_dir, "rev-parse", "HEAD")
        tracked_changes = _git_output(source_dir, "status", "--porcelain", "--untracked-files=no")
        if tracked_changes:
            raise RuntimeError(f"FlashKDA checkout has tracked modifications:\n{tracked_changes}")

        source_package_path = source_dir / "flash_kda" / "__init__.py"
        if _sha256(package_path) != _sha256(source_package_path):
            raise RuntimeError(
                "installed flash_kda package does not match its source checkout: "
                f"package={package_path}, source={source_package_path}"
            )

        provenance.update({"source_dir": str(source_dir), "source_commit": source_commit})
        cutlass_dir = source_dir / "cutlass"
        if cutlass_dir.is_dir():
            cutlass_commit = _git_output(cutlass_dir, "rev-parse", "HEAD")
            submodule_record = _git_output(source_dir, "ls-tree", "HEAD", "cutlass").split()
            if len(submodule_record) < 3 or submodule_record[2] != cutlass_commit:
                raise RuntimeError("FlashKDA CUTLASS checkout does not match its gitlink")
            provenance["cutlass_commit"] = cutlass_commit
    return flash_kda, provenance


def prepare_flashkda_raw_reference(case: dict[str, Any]) -> FlashKDARawReference:
    """Prepare and validate the installed raw FlashKDA peer."""
    flash_kda, provenance = _load_flash_kda_peer()
    cfg = case["config"]
    batch = 1 if cfg.packed else cfg.num_seqs
    seq_len = cfg.total_tokens if cfg.packed else cfg.seq_lens[0]

    def reshaped(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(batch, seq_len, cfg.num_heads, -1)

    peer_out = torch.empty_like(reshaped(case["out"]))
    peer_initial_state = case["initial_state"].clone() if cfg.use_initial_state else None
    peer_final_state = torch.empty_like(case["final_state"]) if cfg.store_final_state else None
    workspace_bytes = flash_kda.get_workspace_size(cfg.total_tokens, cfg.num_heads, cfg.num_seqs)
    workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=case["q"].device)

    def launch() -> None:
        flash_kda._fwd_raw(
            reshaped(case["q"]),
            reshaped(case["k"]),
            reshaped(case["v"]),
            reshaped(case["g"]),
            case["beta"].reshape(batch, seq_len, cfg.num_heads),
            case["scale"],
            peer_out,
            workspace,
            case["A_log"],
            case["dt_bias"],
            cfg.lower_bound,
            initial_state=peer_initial_state,
            final_state=peer_final_state,
            cu_seqlens=case["cu_seqlens"] if cfg.packed else None,
        )

    launch()
    torch.cuda.synchronize()
    expected_out = case["out"]
    actual_out = peer_out.reshape_as(expected_out)
    out_max_abs = float((actual_out.float() - expected_out.float()).abs().max())
    torch.testing.assert_close(actual_out, expected_out, rtol=1e-2, atol=1e-2)

    correctness = {"output_max_abs": out_max_abs}
    if cfg.store_final_state:
        if peer_final_state is None:
            raise AssertionError("FlashKDA raw final-state buffer was not prepared")
        state_max_abs = float((peer_final_state.float() - case["final_state"].float()).abs().max())
        torch.testing.assert_close(peer_final_state, case["final_state"], rtol=1e-2, atol=1e-2)
        correctness["final_state_max_abs"] = state_max_abs

    provenance.update(
        {
            "timing_scope": "flash_kda._fwd_raw",
            "workspace_bytes": int(workspace_bytes),
            "use_initial_state": cfg.use_initial_state,
            "store_final_state": cfg.store_final_state,
        }
    )
    return FlashKDARawReference(launch=launch, provenance=provenance, correctness=correctness)
