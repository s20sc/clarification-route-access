"""Bind anonymous upper-box choices to receipt-verified B1K action suffixes."""
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
import runpy


PROJECT = Path(__file__).resolve().parents[2]
AUDIT = PROJECT / "audit"
CONTRACT = runpy.run_path(str(PROJECT / "research/design-59/retrieval_contract.py"))
BOXES = {name: f"storage_container.n.01_{i}" for i, name in enumerate(CONTRACT["BOXES"], 1)}
ALIASES = ("A", "B")


def read(path):
    return json.loads(path.read_bytes())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def freeze_window(instance, left, right):
    paths = {}
    receipts = {}
    terminals = {}
    for plan, design in zip(("box1_bottom", "box2_bottom"), (left, right)):
        audit = AUDIT / design
        record = read(audit / "reconciled-terminal.json")
        result = record["result"] if "result" in record else record["episodes"][0]["result"]
        assert result["passed"] and result["plan"] == plan and result["consecutive_stable_steps"] >= 30
        path = audit / "physical-results/probe" / str(instance) / plan
        paths[plan] = path
        receipts[plan] = {}
        for kind in ("issued", "completed"):
            file = path / f"{kind}.jsonl"
            data = file.read_bytes()
            assert sha(data) == record["source_snapshot"]["files"][str(file.relative_to(audit / "physical-results"))]["sha256"]
            receipts[plan][kind] = data.splitlines(keepends=True)
        assert all(len(rows) == result["completed"] for rows in receipts[plan].values())
        terminals[plan] = dict(actions=result["completed"], first_official=result["first_official_step"])
    p1, p2 = paths.values()
    for name in ("window.json", "initial-serialized.npy"):
        assert (p1 / name).read_bytes() == (p2 / name).read_bytes()

    lengths = {}
    hashes = {}
    for kind in ("issued", "completed"):
        one, two = (r[kind] for r in receipts.values())
        length = next(i for i, (a, b) in enumerate(zip(one, two)) if a != b)
        assert length > 0
        lengths[kind] = length
        hashes[kind] = sha(b"".join(one[:length]))
    assert lengths["issued"] == lengths["completed"]
    fork = lengths["issued"]
    common = json.loads(receipts["box1_bottom"]["completed"][fork-1])
    assert common["index"] == fork and not any(common["official"].values())
    routes = {}
    for plan, route in receipts.items():
        routes[plan] = dict(
            trace_directory=str(paths[plan].relative_to(PROJECT)), end=terminals[plan],
            action_suffix_sha256=sha(b"".join(route["issued"][fork:])),
            observation_suffix_sha256=sha(b"".join(route["completed"][fork:])),
            remaining_actions=len(route["issued"]) - fork,
        )
    return dict(instance=instance, after_action=fork, common=common, common_hashes=hashes, routes=routes)


def public_observation(window, alias_to_box, order, public_context):
    """Expose only the selector's view, with no route IDs or hidden demand."""
    assert set(alias_to_box) == set(ALIASES) and set(alias_to_box.values()) == set(BOXES)
    assert set(order) == set(ALIASES) and len(order) == len(ALIASES)
    common = window["common"]
    box_to_alias = {box: alias for alias, box in alias_to_box.items()}
    held = set(common["held_by_arm"].values())
    options = []
    for alias in order:
        box = alias_to_box[alias]
        other, = set(ALIASES) - {alias}
        if held:
            operation = ("Stage the held box, deliver the other box, then retrieve and stack the held box."
                         if BOXES[box] in held else "Deliver the held box to the garage floor and stack the other box on it.")
        else:
            operation = f"Deliver {other} to the garage floor, then stack {alias} on it."
        options.append(dict(upper=alias, operation=operation))
    info = dict(
        boxes=[dict(label=alias, center_xyz_m=[round(x, 3) for x in common["box_pose"][BOXES[alias_to_box[alias]]][0]],
                    held=BOXES[alias_to_box[alias]] in held) for alias in ALIASES],
        robot_position_m=[round(x, 3) for x in common["base_pose"][0]],
        plans=options,
    )
    ctx = {key: public_context[key] for key in ("question_cost", "cue_reliability")}
    for key in ("visible_preference", "cue"):
        val = public_context[key]
        ctx[key] = box_to_alias[val] if val is not None else None
    return dict(scene=info, request=ctx)


def dispatch(window, alias_to_box, trial, decision):
    """One clarification at most. A legal final answer binds exactly one frozen suffix."""
    action = "ask" if decision == "ask" else alias_to_box.get(decision) if isinstance(decision, str) else None
    event = trial.submit(action)
    if event["event"] == "user_reply":
        return dict(event="user_reply", requested_upper=next(
            alias for alias, box in alias_to_box.items() if box == event["top_requested"]))
    plan = event["plan"]
    route = window["routes"][plan]
    return dict(event="physical_dispatch", instance=window["instance"], after_action=window["after_action"],
                plan=plan, prefix_sha256=window["common_hashes"]["issued"],
                suffix_sha256=route["action_suffix_sha256"], expected_observation_suffix_sha256=route["observation_suffix_sha256"],
                remaining_actions=route["remaining_actions"])


def check():
    windows = {
        301: freeze_window(301, "design-58", "design-57"),
        314: freeze_window(314, "design-85", "design-66"),
    }
    assert [w["after_action"] for w in windows.values()] == [862, 1170]
    private = dict(visible_preference=None, cue="box1", cue_reliability="3/4",
                   question_cost="1/8", hidden_preference="box2")
    assert Fraction(private["question_cost"]) < 1 - Fraction(private["cue_reliability"])
    assert CONTRACT["decision_rule"](CONTRACT["public_context"](private)) == "ask"
    checks = 0
    for window, labels, order in itertools.product(windows.values(), itertools.permutations(BOXES), itertools.permutations(ALIASES)):
        mapping = dict(zip(ALIASES, labels))
        ctx = CONTRACT["public_context"](private)
        packet = public_observation(window, mapping, order, ctx)
        reversed_hidden = dict(private, hidden_preference="box1")
        assert public_observation(window, mapping, order, CONTRACT["public_context"](reversed_hidden)) == packet
        assert {x["upper"] for x in packet["scene"]["plans"]} == set(ALIASES)
        trial = CONTRACT["Trial"](private)
        reply = dispatch(window, mapping, trial, "ask")
        assert mapping[reply["requested_upper"]] == "box2"
        bound = dispatch(window, mapping, trial, reply["requested_upper"])
        assert bound["event"] == "physical_dispatch" and bound["plan"] == "box1_bottom"
        try:
            dispatch(window, mapping, trial, "ask")
        except CONTRACT["DecisionError"]:
            pass
        else:
            raise AssertionError("dispatch permitted a second question")
        for alias, box in mapping.items():
            test = CONTRACT["Trial"](private)
            actual = dispatch(window, mapping, test, alias)
            assert actual["plan"] == CONTRACT["BIND"][box]
        checks += 1
    path = AUDIT / "design-86/decision-windows.json"
    output = dict(windows=windows, tested_label_order_pairs=checks, tested_direct_bindings=checks*2,
                  tested_clarification_bindings=checks, model_calls=0,
                  scope="Sealed action/observation receipt bindings; no claim of a model intervention.")
    if path.exists():
        assert json.loads(path.read_bytes()) == json.loads(json.dumps(output))
    else:
        path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(dict(forks={i: w["after_action"] for i, w in windows.items()}, pair_checks=checks)))


if __name__ == "__main__":
    check()
