"""Reconcile the raw model receipts and evaluate the frozen stability criterion."""
from collections import Counter
from fractions import Fraction
import itertools
from pathlib import Path
import json
import runpy

SOURCE = Path(__file__).resolve().parent
LIB = runpy.run_path(str(SOURCE / "run_choice.py"))
AUDIT, ROOT = LIB["AUDIT"], LIB["ROOT"]
read, sha, encode, write = (LIB[key] for key in ("read", "sha", "encode", "write"))
WINDOWS = read(LIB["WINDOWS"])["windows"]
BRIDGE = LIB["BRIDGE"]


def load_lines(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def main():
    matrix = AUDIT / "matrix.json"
    out = AUDIT / "model-results"
    design = read(matrix)
    assert all(sha((ROOT / name).read_bytes()) == h for name, h in design["source_hashes"].items())
    assert read(out / "invocation.json")["matrix_sha256"] == sha(matrix.read_bytes())
    complete = read(out / "complete.json")
    requests = load_lines(out / "requests.jsonl")
    responses = load_lines(out / "responses.jsonl")
    decisions = load_lines(out / "decisions.jsonl")
    assert len(requests) == len(responses) == complete["calls"] <= design["max_total_calls"]
    assert len(decisions) == complete["trials"] == design["max_first_calls"]
    assert complete["model_end_matches"] and complete["physical_dispatches_executed"] == 0
    by_case = {case["id"]: case for case in design["cases"]}
    profiles = design["profiles"]
    entries = {}
    for req, res in zip(requests, responses):
        key = (req["case"], req["profile"], req["stage"])
        assert key not in entries
        assert {k: v for k, v in req.items() if k != "payload"} == {
            k: v for k, v in res.items() if k not in ("decision", "response")}
        assert sha(encode(req["payload"])) == req["payload_sha256"]
        entries[key] = (req["payload"], res)
    counts = {profile: Counter() for profile in profiles}
    expected_gaps = {profile: Fraction(0) for profile in profiles}
    grouped = {}
    pairs = set()
    checked = []
    for row in decisions:
        cid, profile = row["case"], row["profile"]
        assert (cid, profile) not in pairs
        pairs.add((cid, profile))
        case = by_case[cid]
        trial = LIB["TRIAL"](case["private"])
        counts[profile]["assigned"] += 1
        messages = [dict(role="system", content=design["prompt"]+" "+profiles[profile]),
                    dict(role="user", content=encode(case["packet"]).decode())]
        bound = None
        for stage in (1, 2):
            payload, rec = entries.pop((cid, profile, stage))
            assert payload == dict(model=design["model"]["model"], messages=messages, stream=False, think=False,
                                   options=design["options"], format=design["format"])
            if rec["decision"] is not None:
                assert rec["decision"] == LIB["parse"](rec["response"], payload)
            decision = rec["decision"]
            if stage == 1:
                assert row["first"] == decision
            try:
                result = BRIDGE["dispatch"](WINDOWS[case["instance"]], case["map"], trial, decision)
            except ValueError:
                assert "failure" in row
                break
            if result["event"] == "physical_dispatch":
                bound = result
                assert row["final"] == decision
                break
            assert stage == 1 and result["event"] == "user_reply"
            messages += [dict(role="assistant", content='{"action":"ask"}'),
                         dict(role="user", content=encode(result).decode())]
        assert bound == row["dispatch"]
        counts[profile]["questions"] += trial.questions
        counts[profile]["baseline_first_match"] += int(row["first"] == case["baseline"])
        plan = bound["plan"] if bound else None
        grouped.setdefault((case["instance"], case["index"], profile), []).append(plan)
        expected_gap = None
        if bound:
            counts[profile]["valid_bound_routes"] += 1
            chosen_box = case["map"][row["final"]]
            counts[profile]["scripted_requested_top"] += int(chosen_box == case["private"]["hidden_preference"])
            expected_gap = Fraction(case["losses"][row["first"]]) - Fraction(case["losses"][case["baseline"]])
            assert expected_gap >= 0
            expected_gaps[profile] += expected_gap
        else:
            counts[profile]["invalid_or_incomplete"] += 1
        checked.append(dict(case=cid, profile=profile, first=row["first"], final_plan=plan,
                            regret=str(expected_gap) if expected_gap is not None else None))
    assert not entries and len(pairs) == len(by_case)*len(profiles)

    invariants = {}
    for key, plans in grouped.items():
        assert len(plans) == 4
        invariants[key] = plans[0] if len(set(plans)) == 1 else None
    robust = []
    across = []
    for one, two in itertools.combinations(profiles, 2):
        for idx in range(6):
            match = {}
            for instance in WINDOWS:
                direction = tuple(invariants[instance, idx, profile] for profile in (one, two))
                if all(direction) and direction[0] != direction[1]:
                    match[instance] = direction
                    robust.append(dict(instance=instance, context=idx, profiles=[one, two], plans=direction))
            if len(match) == len(WINDOWS) and len(set(match.values())) == 1:
                across.append(dict(context=idx, profiles=[one, two], plans=next(iter(match.values()))))

    summary = dict(calls=complete["calls"], conditions={p: dict(counts[p]) for p in profiles},
                   expected_regret_valid_assignments={p: str(loss) for p, loss in expected_gaps.items()},
                   within_scene_robust_contrasts=robust, cross_scene_robust_contrasts=across,
                   physical_dispatches_executed=0, results=checked,
                   inputs={name: sha((out/name).read_bytes()) for name in
                           ("requests.jsonl", "responses.jsonl", "decisions.jsonl", "complete.json")})
    write(AUDIT / "reconciled-choice.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
