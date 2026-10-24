"""Save one OpenStax Astronomy 2e section outside git.

The live HTML is retrieved after the temporal cutoff, so the manifest marks it
ineligible for the baseline splits. The file is a rights-tracked sample, not
training text.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from nebulalm.data.adapters.openstax import record_from_html
from nebulalm.data.acquire import fetch_text

URL = "https://openstax.org/books/astronomy-2e/pages/1-1-the-nature-of-astronomy"
SECTION_ID = "1-1-the-nature-of-astronomy"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    retrieved_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    html = fetch_text(URL)
    record = record_from_html(
        section_id=SECTION_ID,
        title="1.1 The Nature of Astronomy",
        canonical_url=URL,
        html=html,
        retrieved_at=retrieved_at,
    )
    raw_dir = root / "data" / "raw" / "openstax"
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / f"{SECTION_ID}.html"
    target.write_text(html, encoding="utf-8")
    public = {key: value for key, value in record.items() if key not in {"raw_text", "text"}}
    public["local_raw_path"] = str(target.relative_to(root)).replace("\\", "/")
    public["baseline_eligible"] = False
    public["exclusion_reason"] = "html_retrieved_after_temporal_cutoff"
    destination = root / "manifests" / "sources" / "live_openstax.jsonl"
    destination.write_text(json.dumps(public, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
