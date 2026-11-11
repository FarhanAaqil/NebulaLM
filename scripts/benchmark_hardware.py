"""Hardware preflight: real optimizer steps on synthetic tokens for each model size.

Measures steady-state training tokens per second, peak allocated and reserved
VRAM, checkpoint write time and size, and GPU temperature and clocks before and
after. The numbers replace the blueprint's assumed throughput bands and set
D_cap. Synthetic tokens are used so the benchmark does not depend on the corpus;
throughput does not depend on token values.

    python scripts/benchmark_hardware.py --out reports/hardware_benchmark.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

import torch

from nebulalm.model.config import load_shape
from nebulalm.model.transformer import NebulaLM
from nebulalm.training.checkpoint import save_checkpoint
from nebulalm.training.optimizer import build_optimizer
from nebulalm.utils.io import read_yaml

ROOT = Path(__file__).resolve().parents[1]
SIZES = (("small_4m.yaml", 300), ("smaller_transformer.yaml", 300), ("nebulalm.yaml", 500))
WARMUP_STEPS = 30
# Blueprint D_cap: tokens that fit a 90 GPU-hour budget, given the measured throughput.
BUDGET_GPU_HOURS = 90
# Training tokens per model as multiples of D (blueprint, compute section).
D_MULTIPLES = {"small_4m.yaml": 10.9, "smaller_transformer.yaml": 41.7, "nebulalm.yaml": 31.7}


def gpu_sensors() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,power.draw,clocks_throttle_reasons.active",
             "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True).stdout.strip()
        temperature, clock, power, throttle = [part.strip() for part in out.split(",")]
        return {"temperature_c": float(temperature), "sm_clock_mhz": float(clock), "power_w": power, "throttle_reasons": throttle}
    except (OSError, subprocess.CalledProcessError, ValueError):
        return {}


def bench(config: str, steps: int, vocab: int, microbatch: int, accumulation: int, precision: str) -> dict:
    shape = load_shape(ROOT / "configs" / "model" / config, vocab_size=vocab)
    base = read_yaml(ROOT / "configs" / "train" / "base.yaml")
    device = "cuda"
    torch.manual_seed(0)
    model = NebulaLM(shape, seed=17014).to(device)
    optimizer = build_optimizer(model, {**base["optimizer"], "peak_lr": 3e-4})
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(precision)
    generator = torch.Generator(device=device).manual_seed(0)
    ids = torch.randint(2, vocab, (accumulation, microbatch, shape.context_length), device=device, generator=generator)
    mask = torch.ones_like(ids)
    # Real windows carry segment ids, which force an explicit attention mask; pay that cost here too.
    segments = (torch.arange(shape.context_length, device=device) >= shape.context_length // 2).long()
    segments = segments.expand(microbatch, -1).contiguous()
    torch.cuda.reset_peak_memory_stats()
    before = gpu_sensors()
    timings = []
    for step in range(steps):
        torch.cuda.synchronize()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for micro in range(accumulation):
            with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
                nll, count = model.sequence_nll(ids[micro], mask[micro], segments)
            scaler.scale(nll / (count * accumulation)).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        if step >= WARMUP_STEPS:
            timings.append(time.perf_counter() - started)
    after = gpu_sensors()
    tokens_per_step = accumulation * microbatch * (shape.context_length - 1)
    halves = len(timings) // 2
    first_half = tokens_per_step * halves / sum(timings[:halves])
    second_half = tokens_per_step * (len(timings) - halves) / sum(timings[halves:])
    with tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch) / "bench.pt"
        started = time.perf_counter()
        save_checkpoint(path, {"model": model.state_dict(), "optimizer": optimizer.state_dict()})
        write_seconds = time.perf_counter() - started
        size = path.stat().st_size
    tokens_per_second = tokens_per_step * len(timings) / sum(timings)
    result = {
        "config": config,
        "parameters": model.parameter_count(),
        "precision": precision,
        "microbatch": microbatch,
        "accumulation": accumulation,
        "measured_steps": len(timings),
        "tokens_per_step": tokens_per_step,
        "tokens_per_second": tokens_per_second,
        "tokens_per_second_first_half": first_half,
        "tokens_per_second_second_half": second_half,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "checkpoint_write_seconds": write_seconds,
        "checkpoint_bytes": size,
        "gpu_before": before,
        "gpu_after": after,
    }
    del model, optimizer
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", type=int, default=8000)
    parser.add_argument("--precision", default="fp16")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "hardware_benchmark.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available to this interpreter")
    base = read_yaml(ROOT / "configs" / "train" / "base.yaml")
    rows = [bench(config, steps, args.vocab, int(base["microbatch"]), int(base["accumulation"]), args.precision)
            for config, steps in SIZES]
    # D_cap: largest D such that every model's planned token multiples fit the budget.
    seconds = sum(D_MULTIPLES[row["config"]] / row["tokens_per_second"] for row in rows)
    d_cap = BUDGET_GPU_HOURS * 3600 / seconds
    report = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "vocab_size": args.vocab,
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "models": rows,
        "d_cap_tokens_at_90_gpu_hours": int(d_cap),
        "gpu_hours_per_million_D": seconds * 1e6 / 3600,
        "note": "Synthetic token ids; throughput is independent of token values. D_cap uses the blueprint's per-model D multiples.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({row["config"]: round(row["tokens_per_second"]) for row in rows}), "D_cap", int(d_cap))


if __name__ == "__main__":
    main()
