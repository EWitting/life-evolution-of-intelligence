"""Mirror check: a text check of a stage's wiring for what the eye catches in the book's diagrams: asymmetries
between mirrored cells (good and bad, taste and pain), relays one side uses and the other bypasses, unread signals.

    python -m life.mirror_check <key>      one stage: changes from its parent, mirrored cells' inputs side by side,
                                           the flags (unexplained ones first)
    python -m life.mirror_check --chain    the unexplained flags of every stage the book's pages show

Mirrors are names that differ only in `MIRROR` tokens (valence_app / valence_av, us_taste / us_pain, da / da_neg,
...). A projection whose mirror is missing or differs (another rule, teacher, sign of the source) is flagged, as is a
modulator that no projection or receptor reads and a region whose output goes nowhere. A flag that is meant goes into
`KNOWN` with its reason; tests/test_mirror_check.py fails on any other flag on the chain (and on a stale entry), so a
new asymmetry is fixed or explained before a stage runs. It backs up, and does not replace, reading the report and
the diagram by hand (CLAUDE.md)."""
import re
import sys
from pathlib import Path

# name tokens (split at "_") and whole input features that mirror each other: the good side and the bad side
MIRROR = {"app": "av", "taste": "pain", "us_app": "us_av", "da": "da_neg", "dopamine": "habenula", "d1": "d2",
          "ventral_pallidum": "gp_habenular"}
MIRROR.update({v: k for k, v in list(MIRROR.items())})
WHOLE = ("da", "da_neg", "dopamine", "habenula", "ventral_pallidum", "gp_habenular", "us_app", "us_av")

# flags that are meant: (flag text, from the report) -> the reason
KNOWN = {
    "no mirror: valence_app -> grasp": "appetite drives the grasp programme; the bad side stops eating through no_feed (1.1)",
    "no mirror: valence_av -> no_feed": "appetite drives the grasp programme; the bad side stops eating through no_feed (1.1)",
    "no mirror: valence_av -> out": "the bad side turns away; the good side acts through grasp (1.1)",
    "no mirror: hungry -> valence_app": "hunger raises appetite only (drives, 1.7)",
    "no mirror: in[taste] -> fed": "satiety is fed by taste only (affect, 1.8)",
    "differs: pallium -> value_app": "both value cells learn from `da`; their D1/D2 rules differ in sign (abcd)",
    "differs: pallium -> value_av": "both value cells learn from `da`; their D1/D2 rules differ in sign (abcd)",
    "no mirror: habenula -> dopamine": "one dopamine population signals both: bad news reaches it through the habenula",
    "unread: us_av": "the pain teacher, replaced by `da_neg` in 2.3c; us_pain still drives `alarm`",
    # learned from pain only (ADR-024, 1.5): learning what is good was to wait for dopamine; from 2.3s on dopamine
    # teaches both sides (appetite_learning)
    "differs: in[vis*.app*] -> valence_app": "1.5-2.2 learn from pain only (ADR-024); 2.3s teaches both sides",
    "differs: in[vis*.app*] -> valence_av": "1.5-2.2 learn from pain only (ADR-024); 2.3s teaches both sides",
    "no mirror: pallium -> valence_av": "2.2 learns from pain only (ADR-024); 2.3s teaches both sides",
    # hands and the actor (2.5)
    "no mirror: valence_app -> ingest": "appetite for the held item drives ingest; the bad side stops it through no_eat (as grasp, no_feed)",
    "no mirror: valence_av -> no_eat": "appetite for the held item drives ingest; the bad side stops it through no_eat (as grasp, no_feed)",
    "no mirror: striatum_d1 -> gpi": "Go inhibits gpi directly; NoGo reaches it through gpe (Dale's law: striatal cells inhibit)",
    "no mirror: striatum_d2 -> gpe": "Go inhibits gpi directly; NoGo reaches it through gpe (Dale's law: striatal cells inhibit)",
    "no mirror: in[held] -> striatum_d1": "a start strength: a full hand releases USE now and then, so putting down can be learned (2.5)",
}


def mirror(name: str) -> str:
    if name in WHOLE or name in ("taste", "pain"):
        return MIRROR.get(name, name)
    return "_".join(MIRROR.get(t, t) for t in name.split("_"))


def label(p) -> str:
    sel = f"[{','.join(p.src_select)}]" if p.src_select else ""
    return f"{p.src}{sel} -> {p.dst}"


def kind(p, signs: dict) -> str:
    """What a projection is, in the terms a diagram shows: its rule and teacher, the sign of its source, slow."""
    k = "fixed" if p.rule == "fixed" else f"{p.rule}({p.modulator or '-'})"
    if p.rule == "fixed" and p.evolve:
        k = "evolved"
    return k + (" inh" if signs.get(p.src) == "inh" else "") + (" slow" if p.lag else "")


def mirrored_kind(p, signs: dict) -> str:
    k = kind(p, signs)
    return re.sub(r"\(([^)]*)\)", lambda m: f"({mirror(m.group(1))})", k)


def flags(brain) -> list[str]:
    signs = {r.name: r.sign for r in brain.regions}
    names = set(signs) | {"in", "out"}
    proj = {(p.src, p.src_select, p.dst): p for p in brain.projections}
    out = []
    for (src, sel, dst), p in proj.items():
        m = (mirror(src), tuple(mirror(s) for s in sel), mirror(dst))
        if m == (src, sel, dst):
            continue
        if not all(n in names for n in (m[0], m[2])):
            continue
        q = proj.get(m)
        if q is None:
            out.append(f"no mirror: {label(p)}")
        elif mirrored_kind(p, signs) != kind(q, signs):
            out.append(f"differs: {label(p)}")
    mods = {m.name for m in brain.modulators}
    read = {p.modulator for p in brain.projections if p.modulator}
    read |= {rec[0] for r in brain.regions for rec in r.receptors}
    read |= {t for m in brain.modulators for t, _ in m.terms}
    out += [f"unread: {m}" for m in sorted(mods - read)]
    sources = {p.src for p in brain.projections} | {m.pos for m in brain.modulators} | {m.neg for m in brain.modulators}
    sources |= {t for m in brain.modulators for t, _ in m.terms}
    out += [f"dead end: {r.name}" for r in brain.regions if r.name not in sources]
    return out


def report(key: str) -> str:
    from life.experiments.stages import STAGES
    s = STAGES[key]
    b = s.brain
    signs = {r.name: r.sign for r in b.regions}
    lines = [f"stage {key} ({s.name}), parent {s.parent}"]
    if s.parent:
        pb = STAGES[s.parent].brain
        psigns = {r.name: r.sign for r in pb.regions}
        old = {label(p): kind(p, psigns) for p in pb.projections}
        new = {label(p): kind(p, signs) for p in b.projections}
        oldr, newr = {r.name for r in pb.regions}, set(signs)
        lines.append("changes from the parent:")
        lines += [f"  + region {n}" for n in sorted(newr - oldr)] + [f"  - region {n}" for n in sorted(oldr - newr)]
        lines += [f"  + {k}  {new[k]}" for k in new if k not in old]
        lines += [f"  - {k}  {old[k]}" for k in old if k not in new]
        lines += [f"  ~ {k}  {old[k]} -> {new[k]}" for k in new if k in old and old[k] != new[k]]
    lines.append("mirrored cells, inputs side by side:")
    done = set()
    for r in b.regions:
        m = mirror(r.name)
        if m == r.name or m not in signs or r.name in done:
            continue
        done |= {r.name, m}
        ins = {n: sorted(f"{label(p).split(' -> ')[0]}  {kind(p, signs)}" for p in b.projections if p.dst == n)
               for n in (r.name, m)}
        lines.append(f"  {r.name:24s} | {m}")
        for i in range(max(len(ins[r.name]), len(ins[m]))):
            a = ins[r.name][i] if i < len(ins[r.name]) else ""
            c = ins[m][i] if i < len(ins[m]) else ""
            lines.append(f"    {a:38s} | {c}")
    fl = flags(b)
    lines.append("flags (unexplained first):")
    lines += [f"  ! {f}" for f in fl if f not in KNOWN]
    lines += [f"    {f}  -- {KNOWN[f]}" for f in fl if f in KNOWN]
    return "\n".join(lines)


BUILDING = ()   # stages being designed, not yet in the book: checked as the chain is


def chain() -> list[str]:
    """The stages the book's pages show (as tools/brain_layout), and those being built."""
    keys = set(BUILDING)
    for f in (Path(__file__).resolve().parents[1] / "book").glob("*.qmd"):
        keys |= set(re.findall(r'data-stage="([^"]+)"', f.read_text(encoding="utf-8")))
    return sorted(keys)


if __name__ == "__main__":
    if sys.argv[1:] == ["--chain"]:
        from life.experiments.stages import STAGES
        for k in chain():
            fl = [f for f in flags(STAGES[k].brain) if f not in KNOWN]
            print(f"{k}: " + ("ok" if not fl else "; ".join(fl)))
    else:
        print(report(sys.argv[1]))
