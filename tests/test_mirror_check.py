"""Every asymmetry between mirrored cells on the book's chain is fixed or explained (life/mirror_check.py)."""
from life import mirror_check as M
from life.experiments.stages import STAGES


def test_no_unexplained_flags():
    bad = {k: [f for f in M.flags(STAGES[k].brain) if f not in M.KNOWN] for k in M.chain()}
    assert not any(bad.values()), f"explain in mirror_check.KNOWN or fix: {bad}"


def test_known_is_not_stale():
    seen = {f for k in M.chain() for f in M.flags(STAGES[k].brain)}
    assert not set(M.KNOWN) - seen, f"no longer flagged, remove from KNOWN: {set(M.KNOWN) - seen}"
