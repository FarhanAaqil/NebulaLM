"""Copy each finished run's summary into the repository (reports/training/runs.json).

Checkpoints stay in NEBULALM_WORK_DIR; this file keeps the numbers, configs'
hashes, and final-checkpoint hashes in git so they survive the work folder.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("NEBULALM_WORK_DIR", ROOT))


def main() -> None:
    rows = {}
    for run in sorted((WORK / "runs").iterdir()):
        result_path = run / "result.json"
        if not result_path.exists():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        config = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
        final = sorted((run / "checkpoints").glob("step_*.pt.sha256"))
        rows[run.name] = {
            "model": config["identity"]["model"]["name"],
            "parameters": config.get("parameters"),
            "peak_lr": config["identity"]["peak_lr"],
            "seed": config["identity"]["seed"],
            "schedule": config["schedule"],
            "steps_per_pass": config["steps_per_pass"],
            "final_step": result["final_step"],
            "passes": result["passes"],
            "tokens_seen": result["tokens_seen"],
            "skipped_steps": result["skipped_steps"],
            "final_validation_bpb": result["history"][-1]["bits_per_byte"] if result["history"] else None,
            "validation_history": [{"step": h["step"], "bpb": h["bits_per_byte"]} for h in result["history"]],
            "branch_from": config.get("branch_from"),
            "git_commit": config["git"]["commit"],
            "config_sha256": config["config_sha256"],
            "final_checkpoint_sha256": final[-1].read_text(encoding="ascii").strip() if final else None,
        }
    out = ROOT / "reports" / "training" / "runs.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{len(rows)} finished runs -> {out}")


if __name__ == "__main__":
    main()
