"""Reclaim disk from runs already on disk, using the same rules the trainer now applies.

Finished runs keep their final checkpoint and, for trunks, the branch-point
checkpoints. Sweep and proxy runs (lr-*, exp-tok-*) keep result, config, and
logs only: their numbers are recorded and no later stage loads their weights.
Unfinished runs keep their newest valid checkpoint. best.pt and stale .tmp
files are removed everywhere.

    python scripts/prune_runs.py            # report what would be removed
    python scripts/prune_runs.py --apply
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from nebulalm.training.checkpoint import step_path

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("NEBULALM_WORK_DIR", ROOT))


def plan(run: Path) -> list[Path]:
    folder = run / "checkpoints"
    if not folder.exists():
        return []
    files = sorted(folder.iterdir())
    drop = [path for path in files if path.name.startswith("best.pt") or path.suffix == ".tmp"]
    steps = sorted(folder.glob("step_*.pt"))
    if (run / "result.json").exists():
        if run.name.startswith(("lr-", "exp-tok-")):
            keep: set[Path] = set()
        else:
            config = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
            keep = {steps[-1]} if steps else set()
            if run.name.endswith("-trunk"):
                keep |= {step_path(run, int(step)) for step in config["branch_steps"].values()}
    else:
        keep = {steps[-1]} if steps else set()
    for path in steps:
        if path not in keep:
            drop += [path, path.with_name(path.name + ".sha256")]
    return [path for path in drop if path.exists()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    total = 0
    for run in sorted((WORK / "runs").iterdir()):
        if not run.is_dir() or run.name == "pipeline":
            continue
        drop = plan(run)
        size = sum(path.stat().st_size for path in drop)
        total += size
        if drop:
            print(f"{run.name}: {len(drop)} files, {size / 2**30:.2f} GiB")
        if args.apply:
            for path in drop:
                path.unlink()
    print(f"{'removed' if args.apply else 'would remove'} {total / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
