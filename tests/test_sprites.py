"""Dashboard sprites: dressed player bodies. Run: .venv\\Scripts\\python.exe -m pytest -q tests/test_sprites.py"""
import pytest

from life import sprites as S

pytestmark = pytest.mark.skipif(not S.sprites_available(), reason="OHOL sprites not checked out")


def test_wardrobe_items_are_clothing_of_their_slot():
    for slot, ids in S.WARDROBE.items():
        for oid in ids:
            text = S._obj_text(oid, S.DEFAULT_DATA_DIR)
            assert text is not None, oid
            assert S.object_field(text, "clothing") == ("t" if slot == "dress" else slot), oid


def test_dressed_body_differs_and_keeps_its_size():
    pid = S.person_ids()[0]
    naked, dressed = S.render_person(pid), S.render_person(pid, {"t": 2882, "b": 2881, "s": 2887, "h": 1117})
    assert naked is not None and dressed is not None
    assert naked.tobytes() != dressed.tobytes()
    assert abs(dressed.height - naked.height) <= 0.25 * naked.height   # a hat adds a little, nothing lands far away


def test_outfits_are_fixed_and_varied():
    assert S.outfit_for(3) == S.outfit_for(3)
    assert len({tuple(sorted(S.outfit_for(i).items())) for i in range(32)}) > 25
