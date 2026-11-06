"""Verify the controller evidence archive and recheck selected archived mechanism claims."""
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads((ROOT / path).read_text())


def probe(design, instance, route):
    return f"audit/design-{design}/physical-results/probe/{instance}/{route}"


def actions(path):
    return [json.loads(line) for line in (ROOT / path / "follow-up-actions.jsonl").read_text().splitlines()]


def main():
    if not __debug__:
        raise SystemExit("Run with assertions enabled.")
    inventory = read("provenance.json")["files"]
    for path, entry in inventory.items():
        assert sha256((ROOT / path).read_bytes()).hexdigest() == entry["archive_sha256"], path
        if entry["transformation"] == "byte-identical":
            assert entry["source_sha256"] == entry["archive_sha256"], path

    for route in ("box1_bottom", "box2_bottom"):
        start = probe(86, 314, route)
        screens = read(f"{start}/access-candidates.json")
        assert len(screens) == 4 and not actions(start)
        assert all(not item["hover_screen"]["passed"] and item["hover_screen"]["first"]["sample"] == 0
                   for item in screens)
        departed = actions(probe(89, 314, route))
        assert departed and {row["phase"] for row in departed} == {"departure"}
        assert not any(row["held_target"] for row in departed)

    attached = probe(91, 314, "box1_bottom")
    records = actions(attached)
    lift = read(f"{attached}/access-lift-screen.json")
    assert any(row["held_target"] for row in records) and not any(row["phase"] == "lift" for row in records)
    assert not lift["passed"] and lift["first"]["sample"] == 0
    assert any(link == "held_cargo" and "storage_box_80" in body for link, body in lift["first"]["hits"])

    assert not actions(probe(94, 301, "box1_bottom"))
    for design, instance, route in ((93, 314, "box2_bottom"), (95, 301, "box1_bottom")):
        path = probe(design, instance, route)
        records = actions(path)
        candidates = read(f"{path}/access-candidates.json")
        assert len(records) == 114 and {row["phase"] for row in records} == {"departure"}
        assert len(candidates) == 4 and all(not item["hover_screen"]["passed"] for item in candidates)

    print(f"{len(inventory)} evidence digests verified.")
    print("D86 initial screens, D89 departure, D91 attachment/lift stop, D94 zero-action lower and final D93/D95 lower hover failures verified.")


if __name__ == "__main__":
    main()
