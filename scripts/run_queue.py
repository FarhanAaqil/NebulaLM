"""Run the registered training queue in order. Safe to stop and restart at any time.

Each entry is a scripts/train.py invocation with its own run folder. A run with
result.json is finished and skipped; an interrupted run resumes from its newest
valid checkpoint. Learning rates for the grid come from the EXP-LR stage, chosen
on validation by the registered rule, and are written to
experiments/records/EXP-LR.json before any grid run starts.

    python scripts/run_queue.py            # run everything still pending
    python scripts/run_queue.py --list     # print the plan and its status

Tiers follow the blueprint. P2 and P3 items are cut (decision 005).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from nebulalm.utils.io import read_json, read_yaml, write_json

ROOT = Path(__file__).resolve().parents[1]
# Runs and shards are large; they live outside the repository (and outside any synced folder).
WORK = Path(os.environ.get("NEBULALM_WORK_DIR", ROOT))
RUNS = WORK / "runs"
SIZES = {"4m": "small_4m.yaml", "10m": "smaller_transformer.yaml", "29m": "nebulalm.yaml"}
LR_RECORD = ROOT / "experiments" / "records" / "EXP-LR.json"
# Create this file to stop the queue cleanly after the run in progress finishes.
STOP_FILE = WORK / "runs" / "STOP"


def entry(name: str, size: str, *, lr: float | str, shards: str = "m0", passes: float | None = None,
          trunk: bool = False, branch_from: str | None = None, seed: int | None = None, tier: str, experiment: str) -> dict:
    return {"name": name, "size": size, "lr": lr, "shards": shards, "passes": passes, "trunk": trunk,
            "branch_from": branch_from, "seed": seed, "tier": tier, "experiment": experiment}


def plan() -> list[dict]:
    base = read_yaml(ROOT / "configs" / "train" / "base.yaml")
    seeds = read_yaml(ROOT / "configs" / "seeds.yaml")
    branches = base["schedule"]["branch_passes"]
    extra_seeds = [int(seeds["model_init"]) + 1000, int(seeds["model_init"]) + 2000]
    rows: list[dict] = []
    # EXP-LR (P0): every size, three peak LRs, half a pass each (1.5 D per size).
    for size in ("10m", "29m", "4m"):
        for lr in base["lr_candidates"]:
            rows.append(entry(f"lr-{size}-{lr:g}", size, lr=lr, passes=0.5, tier="P0", experiment="EXP-LR"))
    # EXP-GRID (P0): M0 trunk to 6.4 passes and branches at 1, 2, 4, 8, for 10M and 29M.
    for size in ("10m", "29m"):
        rows.append(entry(f"m0-{size}-trunk", size, lr=f"best:{size}", trunk=True, tier="P0", experiment="EXP-GRID"))
        for passes in branches:
            rows.append(entry(f"m0-{size}-e{passes}", size, lr=f"best:{size}", passes=passes,
                              branch_from=f"m0-{size}-trunk", tier="P0", experiment="EXP-GRID"))
    # Branch-equivalence check (G2): a standalone 1-pass run with the trunk's seed and data order.
    rows.append(entry("check-10m-e1-standalone", "10m", lr="best:10m", passes=1, tier="P0", experiment="BRANCH-EQ"))
    # EXP-LEAK (P1): the same grid on M1 (M0 plus probe siblings).
    for size in ("10m", "29m"):
        rows.append(entry(f"m1-{size}-trunk", size, lr=f"best:{size}", shards="m1", trunk=True, tier="P1", experiment="EXP-LEAK"))
        for passes in branches:
            rows.append(entry(f"m1-{size}-e{passes}", size, lr=f"best:{size}", shards="m1", passes=passes,
                              branch_from=f"m1-{size}-trunk", tier="P1", experiment="EXP-LEAK"))
    # 4.5M arm (P1): same grid as 10M on M0.
    rows.append(entry("m0-4m-trunk", "4m", lr="best:4m", trunk=True, tier="P1", experiment="ARM-4M"))
    for passes in branches:
        rows.append(entry(f"m0-4m-e{passes}", "4m", lr="best:4m", passes=passes, branch_from="m0-4m-trunk",
                          tier="P1", experiment="ARM-4M"))
    # EXP-RVU (P1): equal compute; E passes over D/E unique tokens, E in {4, 8}.
    for size in ("10m", "29m"):
        for passes, folder in ((4, "m0_quarter"), (8, "m0_eighth")):
            rows.append(entry(f"rvu-{size}-e{passes}", size, lr=f"best:{size}", shards=folder, passes=passes,
                              tier="P1", experiment="EXP-RVU"))
    # EXP-SEED (P1): two more seeds for 10M and 29M at 1 and 4 passes. The 29M third seed is cut-line 3.
    for size in ("10m", "29m"):
        for seed in extra_seeds:
            rows.append(entry(f"seed{seed}-{size}-e1", size, lr=f"best:{size}", passes=1, seed=seed,
                              tier="P1", experiment="EXP-SEED"))
            rows.append(entry(f"seed{seed}-{size}-e4", size, lr=f"best:{size}", passes=4, seed=seed,
                              tier="P1", experiment="EXP-SEED"))
    return rows


def finished(row: dict) -> bool:
    return (RUNS / row["name"] / "result.json").exists()


def choose_learning_rates(rows: list[dict]) -> dict:
    """Lowest final validation BPB per size; a tie within 1 percent goes to the smaller LR."""
    chosen = {}
    for size in SIZES:
        sweep = [row for row in rows if row["experiment"] == "EXP-LR" and row["size"] == size]
        results = []
        for row in sweep:
            path = RUNS / row["name"] / "result.json"
            if path.exists():
                result = read_json(path)
                results.append({"lr": row["lr"], "val_bpb": result["history"][-1]["bits_per_byte"], "run": row["name"]})
        if len(results) != len(sweep):
            continue
        results.sort(key=lambda item: item["val_bpb"])
        best = results[0]
        close = [item for item in results if (item["val_bpb"] - best["val_bpb"]) / best["val_bpb"] < 0.01]
        pick = min(close, key=lambda item: item["lr"])
        chosen[size] = {"lr": pick["lr"], "results": results,
                        "rule": "lowest validation BPB after half a pass; within 1 percent, the smaller LR"}
    return chosen


def command(row: dict, rates: dict) -> list[str]:
    lr = row["lr"]
    if isinstance(lr, str):
        lr = rates[lr.split(":", 1)[1]]["lr"]
    args = [sys.executable, str(ROOT / "scripts" / "train.py"), "--model", SIZES[row["size"]], "--lr", str(lr),
            "--run", str(RUNS / row["name"]), "--shards", str(WORK / "shards" / row["shards"])]
    if row["trunk"]:
        args.append("--trunk")
    else:
        args += ["--passes", str(row["passes"])]
    if row["branch_from"]:
        args += ["--branch-from", str(RUNS / row["branch_from"])]
    if row["seed"] is not None:
        args += ["--seed", str(row["seed"])]
    return args


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--only", help="run only entries whose name starts with this prefix")
    args = parser.parse_args()
    rows = plan()
    if args.list:
        for row in rows:
            print(f"{'done' if finished(row) else 'todo':4}  {row['tier']}  {row['experiment']:9}  {row['name']}")
        return
    for row in rows:
        if args.only and not row["name"].startswith(args.only):
            continue
        if finished(row):
            continue
        rates = read_json(LR_RECORD) if LR_RECORD.exists() else {}
        if isinstance(row["lr"], str):
            size = row["lr"].split(":", 1)[1]
            if size not in rates:
                rates = choose_learning_rates(rows)
                if size not in rates:
                    raise SystemExit(f"EXP-LR for {size} is not finished; run the sweep first")
                write_json(LR_RECORD, rates)
        if STOP_FILE.exists():
            # Exit non-zero so the pipeline does not mark the queue finished.
            raise SystemExit(f"stopping before {row['name']}: {STOP_FILE} exists (delete it to continue)")
        print("running", row["name"], flush=True)
        subprocess.run(command(row, rates), check=True)


if __name__ == "__main__":
    main()
