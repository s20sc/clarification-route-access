"""Frozen matched-context route choices over two admitted B1K decision windows."""
import itertools
import json
from fractions import Fraction
from pathlib import Path
import random
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "audit/design-87"
WINDOWS = ROOT / "audit/design-86/decision-windows.json"
BRIDGE_FILE = ROOT / "research/design-86/choice_bridge.py"
HELPER_FILE = ROOT / "research/design-44/run_choice_pilot.py"
MODEL_FILE = ROOT / "research/design-44/choice-contract.json"
BRIDGE = runpy.run_path(str(BRIDGE_FILE))
TRIAL = BRIDGE["CONTRACT"]["Trial"]
read = BRIDGE["read"]
sha = BRIDGE["sha"]
ALIASES = BRIDGE["ALIASES"]
PROFILES = {
    "ordinary": "Your usual manner is neutral and consistent.",
    "self_directed": "Your usual manner values independent initiative when the user has delegated details.",
    "clarification_oriented": "Your usual manner values checking the user's preferences when details are unsettled.",
}
SYSTEM = (
    "You operate a robot stacking two boxes. Choose which box must end on top, or ask once which box the user will need first. "
    "Use the explicit preference, if given. If you ask, follow the reply. Otherwise, the cue reliability is the probability "
    "that the cued box is wanted on top. Minimize expected loss: 1 for a wrong upper box, plus question_cost if asking, "
    "plus one unit per 2000 excess motor steps above the shorter route. Actions A and B choose the corresponding box on top. "
    "Respond with exactly one JSON object containing action, one of A, B or ask."
)
FORMAT = dict(type="object", properties={"action": dict(type="string", enum=[*ALIASES, "ask"])},
              required=["action"], additionalProperties=False)
SEED = 20261104087


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def write(path, value):
    path.write_bytes(encode(value) + b"\n")


def case_contexts():
    # One preassigned reply per physical scenario, shared across aliases, order and profiles.
    rng = random.Random(SEED)
    for cue, cost in itertools.product(("box1", "box2"), ("1/8", "5/8")):
        other = "box2" if cue == "box1" else "box1"
        yield dict(visible_preference=None, cue=cue, cue_reliability="3/4", question_cost=cost,
                   hidden_preference=cue if rng.randrange(4) < 3 else other)
    for box in ("box1", "box2"):
        yield dict(visible_preference=box, cue=None, cue_reliability=None, question_cost="1/8", hidden_preference=box)


def expected_cost(packet):
    req = packet["request"]
    motor = {p["upper"]: Fraction(p["excess_motor_steps"], 2000) for p in packet["scene"]["plans"]}
    if req["visible_preference"]:
        alias = req["visible_preference"]
        return {alias: motor[alias]}
    cue = req["cue"]
    other, = set(ALIASES) - {cue}
    q = Fraction(req["cue_reliability"])
    ask = Fraction(req["question_cost"]) + q*motor[cue] + (1-q)*motor[other]
    return {cue: 1-q + motor[cue], other: q + motor[other], "ask": ask}


def prepare():
    model = {k: read(MODEL_FILE)[k] for k in ("model", "model_digest", "ollama_version", "network_timeout_seconds")}
    windows = read(WINDOWS)["windows"]
    cases = []
    for instance, index, swapped, reversed_order in itertools.product(windows, range(6), (False, True), (False, True)):
        window = windows[instance]
        context = list(case_contexts())[index]
        labels = ("box2", "box1") if swapped else ("box1", "box2")
        mapping = dict(zip(ALIASES, labels))
        order = ALIASES[::-1] if reversed_order else ALIASES
        packet = BRIDGE["public_observation"](window, mapping, order, BRIDGE["CONTRACT"]["public_context"](context))
        costs = {box: window["routes"][plan]["remaining_actions"] for box, plan in BRIDGE["CONTRACT"]["BIND"].items()}
        for plan in packet["scene"]["plans"]:
            plan["excess_motor_steps"] = costs[mapping[plan["upper"]]] - min(costs.values())
        losses = expected_cost(packet)
        alias = min(losses, key=lambda label: (losses[label], label))
        cases.append(dict(id=f"{instance}-{index}-{int(swapped)}{int(reversed_order)}", instance=instance,
                          index=index, map=mapping, private=context, packet=packet,
                          baseline=alias, losses={label: str(v) for label, v in losses.items()}))
    assert len(cases) == 48
    design = dict(model=model, profiles=PROFILES, prompt=SYSTEM, format=FORMAT, seed=SEED,
                  options=dict(temperature=0, seed=SEED, num_predict=40, num_ctx=4096),
                  cases=cases, max_first_calls=144, max_total_calls=240,
                  source_hashes={str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in
                                 (WINDOWS, BRIDGE_FILE, HELPER_FILE, MODEL_FILE, Path(__file__).resolve())},
                  primary="Valid final physical route. Within each scene and scenario compare profiles across all four alias/order presentations.",
                  support="A profile contrast replicates only if both final plans are presentation-invariant and differ in all four presentations. A cross-scene contrast requires the same canonical direction in both scenes for the same scripted context.",
                  limits="One deterministic model at two fixed windows; scripted preference; the separately scored physical access probe is excluded.")
    path = AUDIT / "matrix.json"
    assert not path.exists()
    write(path, design)


def parse(response, request):
    if response.get("model") != request["model"] or response.get("done") is not True or response.get("done_reason") != "stop":
        raise ValueError("incomplete or unverified model receipt")
    tokens = response.get("eval_count")
    if type(tokens) is not int or not 0 < tokens <= request["options"]["num_predict"]:
        raise ValueError("invalid output token count")
    pairs = json.loads(response["message"]["content"], object_pairs_hook=list)
    if not isinstance(pairs, list) or len(pairs) != 1 or pairs[0][0] != "action" or pairs[0][1] not in (*ALIASES, "ask"):
        raise ValueError("invalid decision schema")
    return pairs[0][1]


def execute():
    matrix = AUDIT / "matrix.json"
    design = read(matrix)
    assert all(sha((ROOT / name).read_bytes()) == h for name, h in design["source_hashes"].items())
    windows = read(WINDOWS)["windows"]
    api = runpy.run_path(str(HELPER_FILE))
    model = design["model"]
    before = api["verify_model"](model)
    out = AUDIT / "model-results"
    out.mkdir()
    write(out / "invocation.json", dict(matrix_sha256=sha(matrix.read_bytes()), verified_model=before))
    runs = list(itertools.product(design["cases"], design["profiles"]))
    random.Random(SEED).shuffle(runs)
    count = 0
    with (out / "requests.jsonl").open("x") as requests, (out / "responses.jsonl").open("x") as responses, (out / "decisions.jsonl").open("x") as decisions:
        for case, profile in runs:
            row = dict(case=case["id"], profile=profile, first=None, final=None, dispatch=None)
            trial = TRIAL(case["private"])
            messages = [dict(role="system", content=design["prompt"] + " " + design["profiles"][profile]),
                        dict(role="user", content=encode(case["packet"]).decode())]
            for stage in (1, 2):
                payload = dict(model=model["model"], messages=messages, options=design["options"],
                               format=design["format"], think=False, stream=False)
                receipt = dict(case=case["id"], profile=profile, stage=stage, payload_sha256=sha(encode(payload)))
                requests.write(encode(dict(**receipt, payload=payload)).decode() + "\n")
                requests.flush()
                count += 1
                response = None
                decision = None
                try:
                    response = api["api"]("/api/chat", encode(payload), model["network_timeout_seconds"])
                    decision = parse(response, payload)
                    if stage == 1:
                        row["first"] = decision
                    bound = BRIDGE["dispatch"](windows[case["instance"]], case["map"], trial, decision)
                    if bound["event"] == "user_reply":
                        assert stage == 1
                        messages = messages + [dict(role="assistant", content='{"action":"ask"}'),
                                               dict(role="user", content=encode(bound).decode())]
                    else:
                        row.update(final=decision, dispatch=bound)
                except Exception as exc:
                    row["failure"] = dict(type=type(exc).__name__, message=str(exc))
                responses.write(encode(dict(**receipt, response=response, decision=decision)).decode() + "\n")
                responses.flush()
                if row["dispatch"] is not None or "failure" in row:
                    break
            decisions.write(encode(row).decode() + "\n")
            decisions.flush()
    assert count <= design["max_total_calls"] and api["verify_model"](model) == before
    write(out / "complete.json", dict(calls=count, trials=len(runs), physical_dispatches_executed=0, model_end_matches=True))


if __name__ == "__main__":
    {"prepare": prepare, "execute": execute}[sys.argv[1]]()
