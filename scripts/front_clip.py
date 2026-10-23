"""The book's front-page clip: a creature in its world beside its brain, recorded from a published dashboard.

    python scripts/front_clip.py                    # the latest stage of the chain (book/data/growth.json), scene from
                                                    # book/media/front_clip.json, else picked automatically
    python scripts/front_clip.py --pick             # only list good scenes (agent, first tick) and stop
    python scripts/front_clip.py --agent 3 --t0 1200 --ticks 200

The recording is that of the stage's published dashboard (book/dashboards/, `python -m life.book dashboards <key>`),
shown by the current life/dashboard.html in its hero mode (?hero: the world and the brain only, connections glowing
with the activity they carry). Headless Edge (playwright) takes one screenshot per tick and ffmpeg (imageio-ffmpeg)
joins them into book/media/front.mp4, with front.jpg (the first frame) as the poster. Needs the media extra:
uv pip install -e .[media].

A scene picked by hand goes in book/media/front_clip.json ({"stage": "2.3s", "agent": 3, "t0": 1200, ...}); it is
used while its stage is the latest one, so a new stage falls back to the automatic choice until a scene is picked.
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parent.parent
BOOK = ROOT / "book"
MEDIA = BOOK / "media"
TEMPLATE = ROOT / "life" / "dashboard.html"
DEFAULTS = {"ticks": 200, "fps": 10, "width": 1280, "height": 720, "scale": 1.25, "zoom": 7.0, "theme": "light",
            "cap2": "its brain, live"}


def latest_stage() -> dict:
    g = json.loads((BOOK / "data" / "growth.json").read_text(encoding="utf-8"))
    st = g["stages"][-1]
    d = json.loads((BOOK / "data" / f"{st['key']}.json").read_text(encoding="utf-8"))
    dash = next((r["dashboard"] for r in d["runs"]["main"] if r.get("dashboard")), None)
    if not dash:
        sys.exit(f"stage {st['key']} has no published dashboard: python -m life.book dashboards {st['key']}")
    return {"stage": st["key"], "title": st["title"], "dashboard": BOOK / dash}


def hero_page(dashboard: Path, out: Path) -> Path:
    """The current template with the recording of a published dashboard (its `const DATA = ...;` line)."""
    html = dashboard.read_text(encoding="utf-8")
    i = html.index("const DATA = ") + len("const DATA = ")
    data = html[i:html.index(";\n", i)]
    out.write_text(TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/null", data), encoding="utf-8")
    return out


def scenes(H: dict, ticks: int, top: int = 5) -> list[tuple[float, int, int, str]]:
    """Windows of `ticks` ticks in which a creature with a brain view stays alive and does the most: bites that fill
    its stomach, and pain (a bad berry), each counted once per event. Best first."""
    out = []
    for i in H["brainAgents"]:
        s = H["series"][str(i)]
        T = len(s["alive"])
        bites = [t for t in range(1, T) if s["food"][t] > s["food"][t - 1] + 1e-6]   # by grasping or eating
        hurts = [t for t in range(1, T) if s["pain"][t] > 0.05 and s["pain"][t - 1] <= 0.05]
        for t0 in range(0, T - ticks, 20):
            if not all(s["alive"][t0:t0 + ticks]):
                continue
            nb = sum(t0 <= t < t0 + ticks for t in bites)
            nh = sum(t0 <= t < t0 + ticks for t in hurts)
            score = nb + 1.5 * nh + (2 if nb and nh else 0)   # both kinds of event in one window is best
            out.append((score, i, t0, f"{nb} bites, {nh} pains"))
    out.sort(reverse=True)
    best, seen = [], set()
    for sc in out:                                   # one window per creature
        if sc[1] not in seen:
            best.append(sc)
            seen.add(sc[1])
        if len(best) == top:
            break
    return best


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", type=int)
    ap.add_argument("--t0", type=int)
    ap.add_argument("--ticks", type=int)
    ap.add_argument("--fps", type=int)
    ap.add_argument("--theme", choices=("light", "dark"))
    ap.add_argument("--pick", action="store_true", help="list good scenes and stop")
    ap.add_argument("--out", default=str(MEDIA / "front"), help="path without extension (.mp4 and .jpg are written)")
    a = ap.parse_args(argv)
    from playwright.sync_api import sync_playwright
    import imageio_ffmpeg

    st = latest_stage()
    spec_file = MEDIA / "front_clip.json"
    spec = json.loads(spec_file.read_text(encoding="utf-8")) if spec_file.exists() else {}
    if spec.get("stage") != st["stage"]:
        spec = {}                                    # picked for an older stage
    c = {**DEFAULTS, "cap": f"{st['title']}|a creature foraging", **spec,
         **{k: v for k, v in vars(a).items() if v is not None and k in ("agent", "t0", "ticks", "fps", "theme")}}
    with tempfile.TemporaryDirectory() as tmp, sync_playwright() as p:
        page_file = hero_page(st["dashboard"], Path(tmp) / "hero.html")
        b = p.chromium.launch(channel="msedge")
        pg = b.new_page(viewport={"width": c["width"], "height": c["height"]}, device_scale_factor=c["scale"])

        def load(agent=None, t0=0):
            q = {"hero": "", "zoom": c["zoom"], "theme": c["theme"], "cap": c["cap"], "cap2": c["cap2"], "t": t0}
            if agent is not None:
                q["agent"] = agent
            pg.goto(page_file.as_uri() + "?" + urlencode(q))
            pg.wait_for_function("!!window.HERO", timeout=180000)

        if c.get("agent") is None or c.get("t0") is None or a.pick:
            load()
            H = pg.evaluate("({actions: HERO.actions, brainAgents: HERO.brainAgents,"
                            " series: Object.fromEntries(HERO.brainAgents.map(i => [i, HERO.series(i)]))})")
            best = scenes(H, c["ticks"])
            print(f"stage {st['stage']} ({st['title']}), {c['ticks']} ticks: best scenes")
            for score, i, t0, what in best:
                print(f"  --agent {i} --t0 {t0}   {what}")
            if a.pick or not best:
                return
            c["agent"], c["t0"] = best[0][1], best[0][2]
        print(f"recording agent {c['agent']}, ticks {c['t0']}..{c['t0'] + c['ticks'] - 1}")
        load(c["agent"], c["t0"])
        pg.wait_for_timeout(800)                     # sprites decode
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        w, h = round(c["width"] * c["scale"]) // 2 * 2, round(c["height"] * c["scale"]) // 2 * 2
        ff = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "image2pipe",
                               "-framerate", str(c["fps"]), "-i", "-", "-vf", f"scale={w}:{h}", "-c:v", "libx264",
                               "-pix_fmt", "yuv420p", "-crf", "28", "-preset", "slow", "-movflags", "+faststart",
                               str(out.with_suffix(".mp4"))], stdin=subprocess.PIPE)
        for k in range(c["ticks"]):
            pg.evaluate("t => { HERO.setTick(t); return HERO.drawn(); }", c["t0"] + k)
            png = pg.screenshot(type="png")
            if k == 0:                               # the poster, shown until the video plays
                subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "image2pipe", "-i", "-",
                                "-vf", f"scale={w}:{h}", "-q:v", "4", str(out.with_suffix(".jpg"))], input=png, check=True)
            ff.stdin.write(png)
            if k % 20 == 0:
                print(f"  frame {k}/{c['ticks']}", flush=True)
        ff.stdin.close()
        ff.wait()
        b.close()
    mp4 = out.with_suffix(".mp4")
    print(f"wrote {mp4} ({mp4.stat().st_size / 1e6:.1f} MB) and {out.with_suffix('.jpg')}")


if __name__ == "__main__":
    main()
