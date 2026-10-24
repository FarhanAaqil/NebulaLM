import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_environment_report_records_gpu_and_torch():
    report = json.loads((ROOT / "reports" / "environment.json").read_text(encoding="utf-8"))
    assert "RTX 3050" in report["gpu"]["name"]
    assert report["gpu"]["memory_total"]
    assert report["torch"]["version"]
    assert report["torch"]["cuda_available"] is False
