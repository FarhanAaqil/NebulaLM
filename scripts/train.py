"""Train one model: a WSD trunk, a cooldown branch from a trunk, or a plain run.

    # trunk: holds the peak to 0.8 * max(branch_passes) passes, saving the
    # branch points 0.8 * E along the way
    python scripts/train.py --model nebulalm.yaml --lr 3e-4 --run runs/m0-29m-trunk --trunk

    # branch: cooldown from the trunk's 0.8 * E checkpoint, ending at E passes
    python scripts/train.py --model nebulalm.yaml --lr 3e-4 --run runs/m0-29m-e4 --branch-from runs/m0-29m-trunk --passes 4

    # plain run of a fixed number of passes (LR sweep, proxies, cosine check)
    python scripts/train.py --model smaller_transformer.yaml --lr 1e-3 --run runs/lr-10m-1e3 --passes 1

Every run writes run_config.json with the hash chain: shard manifest, tokenizer,
config, and git commit. A resume refuses a checkpoint whose chain differs.
"""

from __future__ import annotations

import argparse
import json
import os
import math
import subprocess
from pathlib import Path

import torch

from nebulalm.model.config import load_shape
from nebulalm.model.transformer import NebulaLM
from nebulalm.training.checkpoint import load_checkpoint, write_json_atomic
from nebulalm.training.data import EpochSampler, ShardWindows
from nebulalm.training.optimizer import build_optimizer, group_report
from nebulalm.training.schedule import branch_schedule
from nebulalm.training.trainer import Trainer, run
from nebulalm.utils.hashing import sha256_file, sha256_text
from nebulalm.utils.io import read_json, read_yaml

ROOT = Path(__file__).resolve().parents[1]
# Runs and shards are large; they live outside the repository (and outside any synced folder).
WORK = Path(os.environ.get("NEBULALM_WORK_DIR", ROOT))


def git_commit() -> dict:
    def call(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"

    return {"commit": call("rev-parse", "HEAD"), "dirty": bool(call("status", "--porcelain", "--untracked-files=no"))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--shards", type=Path, default=WORK / "shards" / "m0")
    parser.add_argument("--val-shards", type=Path, default=WORK / "shards" / "validation")
    parser.add_argument("--passes", type=float)
    parser.add_argument("--trunk", action="store_true")
    parser.add_argument("--branch-from", type=Path)
    parser.add_argument("--schedule", choices=["wsd", "cosine"], default="wsd")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision")
    parser.add_argument("--microbatch", type=int)
    parser.add_argument("--accumulation", type=int)
    parser.add_argument("--max-steps", type=int, help="stop early (benchmarks and smoke runs)")
    args = parser.parse_args()

    base = read_yaml(ROOT / "configs" / "train" / "base.yaml")
    seeds = read_yaml(ROOT / "configs" / "seeds.yaml")
    shard_manifest = read_json(args.shards / "manifest.json")
    vocab = int(shard_manifest["vocab_size"])
    shape = load_shape(ROOT / "configs" / "model" / args.model, vocab_size=vocab)
    seed = int(args.seed if args.seed is not None else seeds["model_init"])
    precision = args.precision or (base["precision"] if args.device.startswith("cuda") else "fp32")
    microbatch = args.microbatch or int(base["microbatch"])
    accumulation = args.accumulation or int(base["accumulation"])

    train = ShardWindows(sorted(args.shards.glob("*.bin")))
    val = ShardWindows(sorted(args.val_shards.glob("*.bin")))
    windows_per_step = microbatch * accumulation
    steps_per_pass = math.ceil(len(train) / windows_per_step)
    rules = base["schedule"]
    warmup = max(1, round(steps_per_pass * float(rules["warmup_fraction_of_one_pass"])))
    longest = max(rules["branch_passes"])
    branch_steps = {passes: branch_schedule(steps_per_pass, passes, warmup)["decay_start"] for passes in rules["branch_passes"]}
    if args.trunk:
        # Warmup, then the peak held until the longest branch point; it never decays itself.
        schedule = {"kind": "wsd", "warmup": warmup, "decay_start": 10**12, "total": 10**12}
        max_steps = branch_steps[longest]
        extra = set(branch_steps.values())
    else:
        if args.passes is None:
            raise SystemExit("--passes is required unless --trunk is given")
        if args.schedule == "wsd":
            schedule = branch_schedule(steps_per_pass, args.passes, warmup, float(rules["decay_fraction"]))
        else:
            total = max(1, round(steps_per_pass * args.passes))
            schedule = {"kind": "cosine", "warmup": max(1, math.ceil(0.01 * total)), "total": total}
        max_steps = schedule["total"]
        extra = set()
    if args.max_steps:
        max_steps = min(max_steps, args.max_steps)

    identity = {
        "shard_manifest_sha256": sha256_file(args.shards / "manifest.json"),
        "val_manifest_sha256": sha256_file(args.val_shards / "manifest.json"),
        "tokenizer_sha256": shard_manifest["tokenizer_sha256"],
        "model": shape.__dict__,
        "seed": seed,
        "peak_lr": args.lr,
        "microbatch": microbatch,
        "accumulation": accumulation,
        "precision": precision,
    }
    config = {
        "identity": identity,
        "schedule": schedule,
        "steps_per_pass": steps_per_pass,
        "max_steps": max_steps,
        "branch_steps": branch_steps,
        "base_config": base,
        "git": git_commit(),
        "lock_sha256": sha256_file(ROOT / "requirements.lock"),
        "torch": torch.__version__,
        "device": args.device,
        "train_windows": len(train),
        "train_target_tokens": train.target_tokens(),
        "branch_from": str(args.branch_from) if args.branch_from else None,
    }
    config["config_sha256"] = sha256_text(json.dumps(config, sort_keys=True, default=str))

    torch.manual_seed(seed)
    model = NebulaLM(shape, seed=seed)
    optimizer = build_optimizer(model, {**base["optimizer"], "peak_lr": args.lr})
    # M1 shard folders record the M0 window count, so shared windows keep their M0 order.
    sampler = EpochSampler(len(train), seed=int(seeds["data_shuffle"]), base_size=shard_manifest.get("base_windows"))
    trainer = Trainer(
        model, optimizer, schedule, train, sampler,
        peak_lr=args.lr, microbatch=microbatch, accumulation=accumulation,
        clip_norm=float(base["clip_norm"]), precision=precision, device=args.device,
    )
    args.run.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.run / "run_config.json", {**config, "parameter_groups": group_report(model),
                                                      "parameters": model.parameter_count()})

    fresh = not any((args.run / "checkpoints").glob("step_*.pt")) if (args.run / "checkpoints").exists() else True
    if args.branch_from and fresh:
        start = branch_steps.get(args.passes)
        if start is None:
            raise SystemExit(f"--passes {args.passes} is not a registered branch point")
        trunk_identity = dict(identity)
        payload = load_checkpoint(args.branch_from / "checkpoints" / f"step_{start:08d}.pt", trunk_identity)
        trainer.branch_from(payload)

    eval_every = max(1, round(steps_per_pass * float(base["eval_every_fraction_of_pass"])))
    checkpoint_every = max(1, round(steps_per_pass * float(base["checkpoint_every_fraction_of_pass"])))
    state = run(trainer, args.run, max_steps=max_steps, identity=identity, config=config, val=val,
                eval_every=eval_every, checkpoint_every=checkpoint_every, extra_checkpoints=extra)
    write_json_atomic(args.run / "result.json", {
        "final_step": state.step, "tokens_seen": state.tokens_seen, "skipped_steps": state.skipped_steps,
        "passes": trainer.sampler.passes(), "best_val_bpb": state.best_val_bpb, "best_step": state.best_step,
        "history": state.history,
    })
    print(json.dumps({"run": str(args.run), "step": state.step, "best_val_bpb": state.best_val_bpb}))


if __name__ == "__main__":
    main()
