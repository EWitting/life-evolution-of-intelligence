"""Brain-diagram layout tool: one grid position per module across the whole chain of book stages.

    .venv/Scripts/python.exe tools/brain_layout/serve.py          # then open http://localhost:8765

Reads the brain of every stage the book's pages show from book/data/*.json (read only), serves tool.html with that data, and writes the
layout the page saves to tools/brain_layout/atlas.json and to book/data/brain_layout.json, which the book's brain
diagrams read (book/js/life-book.js). Standard library only; nothing in life/ is imported.
"""
import json
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DATA_DIR = ROOT / "book" / "data"
ATLAS = HERE / "atlas.json"
BOOK_LAYOUT = DATA_DIR / "brain_layout.json"


def book_stages() -> set[str]:
    """The stages the book's pages show (`data-stage="..."` in book/*.qmd): the chain in use. Data files of stages
    taken off the chain (2.3, 2.3h, ...) stay in book/data but are left out of the layout."""
    keys = set()
    for f in (ROOT / "book").glob("*.qmd"):
        keys |= set(re.findall(r'data-stage="([^"]+)"', f.read_text(encoding="utf-8")))
    return keys


def stages() -> list[dict]:
    """The stages with a brain that the book shows, parents before children (the chain may branch)."""
    found, shown = {}, book_stages()
    for f in sorted(DATA_DIR.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        if "brain" not in d or "stage" not in d or d["stage"]["key"] not in shown:
            continue
        b, s = d["brain"], d["stage"]
        found[s["key"]] = {"key": s["key"], "parent": s.get("parent"), "name": s.get("name", ""),
                           "regions": b["regions"], "projections": b["projections"],
                           "modulators": b.get("modulators", []), "actions": b.get("actions", [])}
    order, seen = [], set()

    def visit(k):
        if k in seen or k not in found:
            return
        seen.add(k)
        visit(found[k]["parent"])
        order.append(found[k])
    # "2.3c" -> [(0, 2), (0, 3), (1, "c")]: numbers and letter suffixes never compare against each other
    for k in sorted(found, key=lambda k: [(0, int(x)) if x.isdigit() else (1, x) for x in re.findall(r"\d+|[a-z]+", k)]):
        visit(k)
    return order


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body: bytes, kind="text/html; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path not in ("/", "/index.html"):
            return self._send(404, b"not found", "text/plain")
        atlas = json.loads(ATLAS.read_text(encoding="utf-8")) if ATLAS.exists() else None
        data = json.dumps({"stages": stages(), "atlas": atlas})
        page = (HERE / "tool.html").read_text(encoding="utf-8").replace("/*DATA*/null", data)
        self._send(200, page.encode("utf-8"))

    def do_POST(self):
        if self.path != "/save":
            return self._send(404, b"not found", "text/plain")
        body = self.rfile.read(int(self.headers["Content-Length"]))
        atlas = json.loads(body)
        text = json.dumps(atlas, indent=1) + "\n"
        ATLAS.write_text(text, encoding="utf-8")
        BOOK_LAYOUT.write_text(text, encoding="utf-8")   # what the book's diagrams read (then quarto render / publish)
        print(f"saved {ATLAS.relative_to(ROOT)} and {BOOK_LAYOUT.relative_to(ROOT)} ({len(atlas.get('cells', {}))} modules)", flush=True)
        self._send(200, b"ok", "text/plain")

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print("stages:", " -> ".join(s["key"] for s in stages()))
    print(f"open http://localhost:{port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
