"""A small local web interface for trying the trained models.

    python scripts/web_ui.py            # then open http://localhost:7860

Pick a model, type the start of a sentence, and the model continues it. Only
Python's standard library is used; nothing leaves this machine. Models load on
first use and stay in memory. Output is model text, not a factual source.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

from nebulalm.model.sampling import generate, load_run
from nebulalm.tokenizer.train import NebulaTokenizer

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("NEBULALM_WORK_DIR", ROOT))
SIZES = {"29m": "29.4M", "10m": "10.3M", "4m": "4.5M"}
PASSES = (8, 4, 2, 1)


def available_models() -> list[dict]:
    """M0 cooldown branches that have a checkpoint, with their measured test BPB when known."""
    models = []
    for size, label in SIZES.items():
        for passes in PASSES:
            name = f"m0-{size}-e{passes}"
            if not list((WORK / "runs" / name / "checkpoints").glob("step_*.pt")):
                continue
            entry = {"id": name, "label": f"{label} parameters, {passes} pass{'es' if passes > 1 else ''}"}
            audit = ROOT / "reports" / "audits" / f"{name}.json"
            if audit.exists():
                learn = json.loads(audit.read_text(encoding="utf-8"))["learn"]
                entry["test_bpb"] = learn.get("test_bpb")
                entry["validation_bpb"] = learn.get("validation_bpb")
            models.append(entry)
    return models


class Engine:
    def __init__(self, device: str) -> None:
        self.device = device
        self.tokenizer = NebulaTokenizer.from_file(ROOT / "artifacts" / "tokenizer" / "selected" / "tokenizer.json", "")
        self.models: dict[str, torch.nn.Module] = {}
        self.lock = threading.Lock()
        self.allowed = {model["id"] for model in available_models()}

    def run(self, request: dict) -> dict:
        name = request.get("model")
        if name not in self.allowed:
            raise ValueError("unknown model")
        prompt = str(request.get("prompt", ""))[:2000]
        samples = max(1, min(int(request.get("samples", 1)), 4))
        options = {
            "max_new": max(1, min(int(request.get("max_new", 150)), 200)),
            "temperature": max(0.05, min(float(request.get("temperature", 0.8)), 2.0)),
            "top_k": max(1, min(int(request.get("top_k", 40)), 500)),
            "greedy": bool(request.get("greedy", False)),
        }
        seed = int(request.get("seed", 17018))
        with self.lock:
            if name not in self.models:
                self.models[name] = load_run(WORK / "runs" / name, self.device)[0]
            model = self.models[name]
            generator = torch.Generator(device=self.device).manual_seed(seed)
            ids = self.tokenizer.encode(prompt)
            outputs = []
            for _ in range(1 if options["greedy"] else samples):
                new = generate(model, ids, self.tokenizer.eos_id, generator=generator, device=self.device, **options)
                outputs.append(self.tokenizer.decode(new))
        return {"prompt": prompt, "outputs": outputs, "prompt_tokens": len(ids)}


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NebulaLM</title>
<style>
  :root { --bg:#f7f7f5; --panel:#ffffff; --ink:#1d1d1b; --muted:#6b6b66; --line:#e2e2dd; --accent:#3b5bdb; --soft:#eef1fd; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#141413; --panel:#1e1e1c; --ink:#ececea; --muted:#a3a39e; --line:#33332f; --accent:#91a7ff; --soft:#232842; }
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink); font:16px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }
  main { max-width: 820px; margin: 0 auto; padding: 32px 16px 64px; }
  h1 { font-size: 1.6rem; margin: 0 0 4px; letter-spacing: -0.01em; }
  .sub { color: var(--muted); margin: 0 0 24px; }
  .panel { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:20px; }
  label { display:block; font-size:.85rem; color:var(--muted); margin-bottom:6px; }
  select, textarea, input[type=number] { width:100%; font:inherit; color:var(--ink); background:var(--bg);
    border:1px solid var(--line); border-radius:8px; padding:10px 12px; }
  textarea { min-height: 96px; resize: vertical; }
  .row { display:grid; grid-template-columns: repeat(4, 1fr); gap:12px; margin-top:16px; }
  @media (max-width: 640px) { .row { grid-template-columns: repeat(2, 1fr); } }
  .chips { display:flex; flex-wrap:wrap; gap:8px; margin:10px 0 0; }
  .chip { border:1px solid var(--line); background:var(--bg); color:var(--ink); border-radius:999px; padding:4px 12px;
    font-size:.85rem; cursor:pointer; }
  .chip:hover { border-color: var(--accent); }
  .actions { display:flex; align-items:center; gap:16px; margin-top:20px; flex-wrap:wrap; }
  button.go { background:var(--accent); color:#fff; border:0; border-radius:8px; padding:10px 22px; font:inherit;
    font-weight:600; cursor:pointer; }
  button.go:disabled { opacity:.6; cursor:wait; }
  .check { display:flex; align-items:center; gap:8px; color:var(--muted); font-size:.9rem; }
  .meta { color:var(--muted); font-size:.85rem; margin-top:8px; min-height:1.3em; }
  .out { margin-top:20px; display:grid; gap:12px; }
  .sample { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:16px 18px; white-space:pre-wrap; }
  .sample b { background:var(--soft); font-weight:600; border-radius:4px; padding:0 2px; }
  .sample .tag { display:block; font-size:.75rem; color:var(--muted); margin-bottom:6px; text-transform:uppercase; letter-spacing:.05em; }
  .note { color:var(--muted); font-size:.8rem; margin-top:28px; }
</style>
</head>
<body>
<main>
  <h1>NebulaLM</h1>
  <p class="sub">A small astronomy language model trained from scratch on Wikipedia. Type the start of a sentence and it writes the rest.</p>
  <div class="panel">
    <label for="model">Model</label>
    <select id="model"></select>
    <div class="meta" id="modelinfo"></div>
    <label for="prompt" style="margin-top:14px">Prompt</label>
    <textarea id="prompt">The Andromeda Galaxy is</textarea>
    <div class="chips" id="chips"></div>
    <div class="row">
      <div><label for="temperature">Temperature</label><input id="temperature" type="number" min="0.05" max="2" step="0.05" value="0.8"></div>
      <div><label for="top_k">Top-k</label><input id="top_k" type="number" min="1" max="500" step="1" value="40"></div>
      <div><label for="max_new">Max new tokens</label><input id="max_new" type="number" min="1" max="200" step="10" value="150"></div>
      <div><label for="samples">Samples</label><input id="samples" type="number" min="1" max="4" step="1" value="2"></div>
    </div>
    <div class="actions">
      <button class="go" id="go">Generate</button>
      <span class="check"><input type="checkbox" id="greedy"><label for="greedy" style="margin:0">Greedy (always the most likely word)</label></span>
    </div>
    <div class="meta" id="status"></div>
  </div>
  <div class="out" id="out"></div>
  <p class="note">The model has 4–30 million parameters and writes fluent but often wrong facts. Do not use its output as a source. Bits per byte (BPB) is the model's measured loss on held-out articles; lower is better.</p>
</main>
<script>
const $ = (id) => document.getElementById(id);
const examples = ["The Andromeda Galaxy is", "Jupiter is the fifth planet", "A neutron star forms when",
  "The Hubble Space Telescope", "Saturn's rings are made of", "In 1969, the Apollo 11 mission"];
let models = [];
function describe() {
  const m = models.find(x => x.id === $("model").value);
  $("modelinfo").textContent = m && m.test_bpb ? `Test score: ${m.test_bpb.toFixed(3)} bits per byte (lower is better)` : "";
}
fetch("/models").then(r => r.json()).then(list => {
  models = list;
  for (const m of list) {
    const o = document.createElement("option");
    o.value = m.id; o.textContent = m.label; $("model").appendChild(o);
  }
  describe();
});
$("model").addEventListener("change", describe);
for (const text of examples) {
  const b = document.createElement("button");
  b.className = "chip"; b.textContent = text;
  b.onclick = () => { $("prompt").value = text; $("prompt").focus(); };
  $("chips").appendChild(b);
}
function escapeHtml(s) { return s.replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c])); }
async function run() {
  const body = {
    model: $("model").value, prompt: $("prompt").value,
    temperature: +$("temperature").value, top_k: +$("top_k").value, max_new: +$("max_new").value,
    samples: +$("samples").value, greedy: $("greedy").checked, seed: Math.floor(Math.random() * 1e9)
  };
  $("go").disabled = true; $("status").textContent = "Writing… (the first use of a model takes a few seconds to load)";
  const started = performance.now();
  try {
    const r = await fetch("/generate", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || r.statusText);
    $("out").innerHTML = data.outputs.map((text, i) =>
      `<div class="sample"><span class="tag">Sample ${i + 1}</span><b>${escapeHtml(data.prompt)}</b>${escapeHtml(text)}</div>`).join("");
    $("status").textContent = `Done in ${((performance.now() - started) / 1000).toFixed(1)} s`;
  } catch (e) {
    $("status").textContent = "Error: " + e.message;
  } finally {
    $("go").disabled = false;
  }
}
$("go").addEventListener("click", run);
$("prompt").addEventListener("keydown", e => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) run(); });
</script>
</body>
</html>
"""


def make_handler(engine: Engine):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: bytes, kind: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif self.path == "/models":
                self._send(200, json.dumps(available_models()).encode("utf-8"), "application/json")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:
            if self.path != "/generate":
                self._send(404, b"not found", "text/plain")
                return
            try:
                length = min(int(self.headers.get("Content-Length", 0)), 100_000)
                result = engine.run(json.loads(self.rfile.read(length) or b"{}"))
                self._send(200, json.dumps(result, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                self._send(400, json.dumps({"error": str(exc)}).encode("utf-8"), "application/json")

        def log_message(self, format: str, *args) -> None:
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    engine = Engine(args.device)
    if not engine.allowed:
        raise SystemExit(f"no trained models found under {WORK / 'runs'}; set NEBULALM_WORK_DIR")
    # Bound to localhost only: the interface is for this machine.
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(engine))
    url = f"http://localhost:{args.port}"
    print(f"NebulaLM is running at {url} ({len(engine.allowed)} models, {args.device}). Press Ctrl+C to stop.")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
