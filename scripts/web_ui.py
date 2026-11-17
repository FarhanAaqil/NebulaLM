"""NebulaLM, an interactive local interface for the trained models.

    python scripts/web_ui.py            # then open http://localhost:7860

Pick a model, type the start of a sentence, and watch the model write the rest
token by token over a live-rendered nebula. Only Python's standard library is
used and the server listens on localhost only. Models load on first use and
stay in memory. Output is model text, not a factual source.
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

from nebulalm.model.sampling import iter_generate, load_run, text_deltas
from nebulalm.tokenizer.train import NebulaTokenizer

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("NEBULALM_WORK_DIR", ROOT))
PAGE_PATH = Path(__file__).with_name("web_ui.html")
SIZES = {"29m": ("29.4M", 29_447_168), "10m": ("10.3M", 10_268_928), "4m": ("4.5M", 4_483_328)}
PASSES = (8, 4, 2, 1)


def available_models() -> list[dict]:
    """M0 cooldown branches that have a checkpoint, with their measured BPB when known."""
    models = []
    for size, (label, parameters) in SIZES.items():
        for passes in PASSES:
            name = f"m0-{size}-e{passes}"
            if not list((WORK / "runs" / name / "checkpoints").glob("step_*.pt")):
                continue
            entry = {"id": name, "size": label, "parameters": parameters, "passes": passes}
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

    def options(self, request: dict) -> tuple[str, str, int, dict, int]:
        name = request.get("model")
        if name not in self.allowed:
            raise ValueError("unknown model")
        prompt = str(request.get("prompt", ""))[:2000]
        options = {
            "max_new": max(1, min(int(request.get("max_new", 150)), 200)),
            "temperature": max(0.05, min(float(request.get("temperature", 0.8)), 2.0)),
            "top_k": max(1, min(int(request.get("top_k", 40)), 500)),
            "greedy": bool(request.get("greedy", False)),
        }
        samples = 1 if options["greedy"] else max(1, min(int(request.get("samples", 1)), 3))
        return name, prompt, samples, options, int(request.get("seed", 17018))

    def model(self, name: str) -> torch.nn.Module:
        if name not in self.models:
            self.models[name] = load_run(WORK / "runs" / name, self.device)[0]
        return self.models[name]

    def stream(self, request: dict):
        """Yield events: one 'start' per sample, a 'text' event per decoded piece, and a final 'done'."""
        name, prompt, samples, options, seed = self.options(request)
        with self.lock:
            model = self.model(name)
            generator = torch.Generator(device=self.device).manual_seed(seed)
            ids = self.tokenizer.encode(prompt)
            for index in range(samples):
                yield {"event": "start", "sample": index}
                tokens = iter_generate(model, ids, self.tokenizer.eos_id, generator=generator, device=self.device, **options)
                count = 0

                def counted():
                    nonlocal count
                    for token in tokens:
                        count += 1
                        yield token

                for piece in text_deltas(counted(), self.tokenizer.decode):
                    yield {"event": "text", "sample": index, "text": piece}
                yield {"event": "end", "sample": index, "tokens": count}
        yield {"event": "done"}


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
                self._send(200, PAGE_PATH.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/models":
                self._send(200, json.dumps(available_models()).encode("utf-8"), "application/json")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:
            if self.path != "/stream":
                self._send(404, b"not found", "text/plain")
                return
            try:
                length = min(int(self.headers.get("Content-Length", 0)), 100_000)
                request = json.loads(self.rfile.read(length) or b"{}")
                engine.options(request)  # validate before committing to a streamed response
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                self._send(400, json.dumps({"error": str(exc)}).encode("utf-8"), "application/json")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                for event in engine.stream(request):
                    self.wfile.write(f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass  # the page stopped listening (Stop button or tab closed)

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
