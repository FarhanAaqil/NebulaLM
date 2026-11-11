"""Run every remaining stage of the study in order, unattended and resumable.

Each stage writes runs/pipeline/<stage>.done when it finishes, so a restart
skips finished stages. Long stages (the training queue) resume internally.
Progress goes to runs/pipeline/pipeline.log.

    python scripts/pipeline.py              # run all pending stages
    python scripts/pipeline.py --until pack # stop after a named stage

The test split is read only by the "campaign" stage, after the whole training
queue has finished (decision 005).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.setdefault("NEBULALM_DATA_DIR", "C:/Users/aaqil/nebulalm-data"))
# Runs and shards are large; they live outside the repository (and outside any synced folder).
# Child scripts inherit this environment variable.
WORK = Path(os.environ.setdefault("NEBULALM_WORK_DIR", str(DATA)))
STATE = WORK / "runs" / "pipeline"
WHEEL = DATA / "torch-2.10.0+cu128-cp312-cp312-win_amd64.whl"
WHEEL_BYTES = 2867409626
ENGLISH_FILES = 41
SELECTED = ROOT / "artifacts" / "tokenizer" / "selected"


def log(message: str) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    with (STATE / "pipeline.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def python(*args: str) -> None:
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    log("run " + " ".join(args))
    subprocess.run([sys.executable, *args], cwd=ROOT, env=env, check=True)


def script(name: str, *args: str) -> None:
    python(str(ROOT / "scripts" / name), *args)


def wait_dataset() -> None:
    english = DATA / "hf" / "20231101.en"
    while len(list(english.glob("*.parquet"))) < ENGLISH_FILES:
        time.sleep(20)
    log("dataset download complete")


def install_torch() -> None:
    """Only the GPU stages need this, so the CPU stages run while the wheel downloads."""
    while not WHEEL.exists() or WHEEL.stat().st_size < WHEEL_BYTES:
        time.sleep(20)
    log("CUDA wheel downloaded")
    check = subprocess.run([sys.executable, "-c", "import torch; print(torch.cuda.is_available())"],
                           capture_output=True, text=True)
    if check.stdout.strip() != "True":
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall", str(WHEEL)], check=True)
    check = subprocess.run([sys.executable, "-c", "import torch; print(torch.__version__, torch.cuda.is_available())"],
                           capture_output=True, text=True, check=True)
    log("torch " + check.stdout.strip())
    if "True" not in check.stdout:
        raise SystemExit("CUDA is still unavailable after installing the wheel")


def audit_all(final: bool) -> None:
    runs = sorted(path for path in (WORK / "runs").iterdir() if (path / "result.json").exists() and not path.name.startswith(("lr-", "exp-tok-")))
    for run in runs:
        if not any((run / "checkpoints").glob("step_*.pt")):
            continue
        target = run / ("audit_test.json" if final else "audit.json")
        if target.exists():
            continue
        if run.name.endswith("-trunk"):
            continue
        script("audit_checkpoint.py", "--run", str(run), *(["--final-evaluation-campaign"] if final else []))


STAGES = [
    ("dataset", wait_dataset),
    ("dump", lambda: script("collect_corpus.py", "dump")),
    ("corpus", lambda: script("build_corpus.py")),
    ("tokenizer_train", lambda: script("tokenizer_experiment.py", "train")),
    ("torch", install_torch),
    ("tokenizer_proxies", lambda: script("tokenizer_experiment.py", "proxies")),
    ("tokenizer_select", lambda: script("tokenizer_experiment.py", "select")),
    ("pack", lambda: script("pack_shards.py", "--tokenizer", str(SELECTED), "--out", str(WORK / "shards"))),
    ("bundle", lambda: script("build_eval_bundle.py", "--tokenizer", str(SELECTED))),
    ("benchmark", lambda: script("benchmark_hardware.py")),
    ("ngram", lambda: script("ngram_baseline.py")),
    ("queue", lambda: script("run_queue.py")),
    ("audit", lambda: audit_all(final=False)),
    ("campaign", lambda: (audit_all(final=True), script("ngram_baseline.py", "--final-evaluation-campaign"))),
    ("analysis", lambda: script("analyze_study.py")),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--until")
    args = parser.parse_args()
    for name, action in STAGES:
        marker = STATE / f"{name}.done"
        if marker.exists():
            continue
        log(f"stage {name} started")
        started = time.time()
        try:
            action()
        except Exception as exc:  # recorded, then re-raised so the process exits non-zero
            log(f"stage {name} FAILED: {exc!r}")
            raise
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                      "seconds": round(time.time() - started)}) + "\n", encoding="utf-8")
        log(f"stage {name} done in {round(time.time() - started)}s")
        if args.until == name:
            break


if __name__ == "__main__":
    main()
