"""Reconcile archived model and simulator receipts without network calls."""
from collections import Counter
import csv
from contextlib import redirect_stdout
from fractions import Fraction
from hashlib import sha256
import io
import json
from math import dist, isclose
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads((ROOT / path).read_bytes())


def lines(path):
    return (ROOT / path).read_bytes().splitlines(keepends=True)


def choices(design):
    audit = ROOT / f"audit/design-{design}"
    reconciler = runpy.run_path(str(ROOT / "research/design-87/reconcile_choice.py"))["main"]

    def check_saved(path, result):
        assert json.loads(path.read_bytes()) == json.loads(json.dumps(result))
        summary.update(result)

    summary = {}
    reconciler.__globals__.update(AUDIT=audit, write=check_saved)
    with redirect_stdout(io.StringIO()):
        reconciler()

    expected_cost = runpy.run_path(str(ROOT / "research/design-87/run_choice.py"))["expected_cost"]
    cases = {c["id"]: c for c in read(audit / "matrix.json")["cases"]}
    profiles = {name: Counter() for name in (*summary["conditions"], "minimum_expected_loss")}
    cells = {}

    def score(case, first, upper, total):
        question = first == "ask"
        req = case["packet"]["request"]
        by_box = {case["map"][p["upper"]]: p["excess_motor_steps"] for p in case["packet"]["scene"]["plans"]}
        mismatch = upper != case["private"]["hidden_preference"]
        total["assigned"] += 1
        total["mismatches"] += mismatch
        total["questions"] += question
        total["excess_steps"] += by_box[upper]
        if req["visible_preference"] is None:
            charge = req["question_cost"]
            total[f"admissible_at_{charge}"] += 1
            total[f"questions_at_{charge}"] += question
        if question:
            total["question_charge"] += Fraction(req["question_cost"])
        total["realized_script_loss"] += mismatch + Fraction(by_box[upper], 2000)
        if question:
            total["realized_script_loss"] += Fraction(req["question_cost"])

    def record(case, profile, first, upper):
        cell, presentation = case["id"].rsplit("-", 1)
        cells.setdefault(cell, {}).setdefault(profile, {})[presentation] = dict(
            first=first if first == "ask" else case["map"][first], final_upper=upper)
        score(case, first, upper, profiles[profile])

    for case in cases.values():
        losses = expected_cost(case["packet"])
        first = min(losses, key=lambda label: (losses[label], label))
        assert first == case["baseline"] and losses == {k: Fraction(v) for k, v in case["losses"].items()}
        upper = case["private"]["hidden_preference"] if first == "ask" else case["map"][first]
        record(case, "minimum_expected_loss", first, upper)
    for row in summary["results"]:
        assert row["final_plan"] in ("box1_bottom", "box2_bottom")
        upper = "box2" if row["final_plan"] == "box1_bottom" else "box1"
        record(cases[row["case"]], row["profile"], row["first"], upper)
    for name, counts in profiles.items():
        if name in summary["conditions"]:
            old = summary["conditions"][name]
            assert counts["assigned"] - counts["mismatches"] == old["scripted_requested_top"]
            assert counts["questions"] == old["questions"]
        counts["realized_script_loss"] = str(counts["realized_script_loss"])
        counts["question_charge"] = str(counts["question_charge"])
    return dict(calls=summary["calls"], script_scores=profiles,
                expected_regret=summary["expected_regret_valid_assignments"],
                cells={k: cells[k] for k in sorted(cells)},
                within_scene=summary["within_scene_robust_contrasts"],
                across_scenes=summary["cross_scene_robust_contrasts"])


def shoe_choices():
    binder = runpy.run_path(str(ROOT / "research/design-43/build_choice_draft.py"))
    bind_response = runpy.run_path(str(ROOT / "research/design-46/run_choice_diagnostic.py"))["result_binding"]
    audit = Path("audit/design-47")
    matrix_path = audit / "matrix.json"
    matrix = read(matrix_path)
    raw = audit / "model-results"
    invocation = read(raw / "invocation.json")
    lineage = read("source-provenance.json")["transformed_inputs"][matrix_path.as_posix()]
    assert invocation["matrix_sha256"] == lineage["source_sha256"]
    assert sha256((ROOT / matrix_path).read_bytes()).hexdigest() == lineage["derivative_sha256"]
    cases = matrix["cases"]
    assert invocation["order"] == [c["run_id"] for c in cases]
    historical, mismatches, conflicts = Counter(), Counter(), {}
    valid = matched = 0
    historical_matrices = {
        i: {c["id"]: c for c in read(path)["cases"]}
        for i, path in ((301, "audit/design-43/draft-choice-matrix.json"),
                        (320, "audit/design-46/choice-matrix.json"))
    }
    issued_rows, response_rows = (lines(raw / f"{name}.jsonl") for name in ("issued", "results"))
    assert len(issued_rows) == len(response_rows) == len(cases) == read(raw / "complete.json")["calls"]
    for case, issued, raw_receipt in zip(cases, issued_rows, response_rows):
        assert json.loads(issued) == dict(id=case["run_id"], request_sha256=case["request_sha256"])
        receipt = json.loads(raw_receipt)
        assert case["run_id"] == receipt["id"]
        assert binder["sha"](binder["canonical"](case["request"])) == case["request_sha256"]
        try:
            binding = bind_response(receipt["response"], case, binder["bind_choice"])
        except ValueError as exc:
            assert receipt["failure"] == dict(type=type(exc).__name__, message=str(exc))
            mismatches[str(exc)] += 1
            binding = None
        assert json.loads(json.dumps(binding)) == receipt["binding"]
        valid += binding is not None
        plan = binding["plan"] if binding else None
        if case["format"] == "opaque":
            original_case = historical_matrices[case["instance"]][case["origin"][0]]
            assert (original_case["request"], original_case["request_sha256"]) == (
                case["request"], case["request_sha256"])
            history = "design-44/model-pilot-results" if case["instance"] == 301 else "design-46/model-results"
            original = read(f"audit/{history}/{case['origin'][0]}.result.json")["binding"]
            historical[case["instance"]] += plan == (original["plan"] if original else None)
        else:
            matched += plan == case["rules"]["follow_advice"]
            if case["conflict"]:
                key = f'{case["instance"]}/{int(case["row_flip"])}'
                conflicts.setdefault(key, {})[case["condition"]] = (
                    plan == (case["least_actions"] if case["condition"] == "self_directed" else case["preference"]))
    assert len(conflicts) == 4 and all(len(v) == 3 for v in conflicts.values())
    return dict(assigned=len(cases), valid=valid, opaque_historical_agreement=historical,
                invalid_reasons=mismatches, semantic_advice_matches=matched, semantic_conflict_cells=conflicts)


def masked_pose(audit, plan, follow, target0, other0, rows):
    key = next(c for c in read(audit / "blind-private-key.json")["cases"] if c["plan"] == plan)
    case = key["case"]
    folder = audit / "blind-materials" / case
    for file, digest in key["hashes"].items():
        if file.endswith(".mp4"):
            excluded = read("source-provenance.json")["excluded_video_sha256"]
            assert excluded[(folder / file).as_posix()] == digest
        else:
            assert sha256((ROOT / folder / file).read_bytes()).hexdigest() == digest
    requested = follow["requested"]
    assert key["requested"] == requested
    target, other = sorted(key["labels"], key=lambda label: key["labels"][label] != requested)
    assert key["labels"][other] == follow["nonrequested"]
    with (ROOT / folder / "positions.csv").open(newline="") as f:
        masked = list(csv.DictReader(f))
    assert len(masked) == len(rows) + 1 == key["actions"] + 1
    stable = longest = 0
    max_rise = max_shift = max_turn = 0.0
    for i, m in enumerate(masked):
        assert int(m["frame"]) == i
        xyz = lambda label: [float(m[f"{label}_{axis}_m"]) for axis in "xyz"]
        t, o = xyz(target), xyz(other)
        t_turn, o_turn = (float(m[f"{label}_turn_rad"]) for label in (target, other))
        if i == 0:
            assert (t, o, t_turn, o_turn) == (target0, other0, 0., 0.)
        else:
            row = rows[i-1]
            assert (t, o, t_turn, o_turn) == (row["target_position"], row["other_position"],
                                             row["target_rotation_rad"], row["other_rotation_rad"])
            rise = t[2] - target0[2]
            max_rise = max(max_rise, rise)
            max_shift = max(max_shift, dist(o, other0))
            max_turn = max(max_turn, o_turn)
            stable = stable + 1 if rise >= follow["rise_m"] else 0
            longest = max(longest, stable)
    pose = "yes" if (longest >= follow["stable_actions"] and max_shift <= follow["other_translation_m"]
                     and max_turn <= follow["other_rotation_rad"]) else "no"
    blind = read(audit / "blind-reconciliation.json")
    expected = blind["expected"][case]
    assert (pose, longest) == (expected["pose_lift"], expected["longest_lift_run"])
    for field, result in (("max_requested_rise_m", max_rise), ("max_other_translation_m", max_shift),
                          ("max_other_rotation_rad", max_turn)):
        assert isclose(result, expected[field], abs_tol=1e-15)
    votes = []
    for file, digest in sorted(blind["report_hashes"].items()):
        path = audit / "blind-reports" / file
        assert sha256((ROOT / path).read_bytes()).hexdigest() == digest
        vote = next(row for row in read(path)["cases"] if row["case_id"] == case)
        votes.append({k: vote[k] for k in ("pose_lift", "visual_lift", "longest_lift_run")})
    return dict(case=case, pose_criterion=pose, longest_pose_run=longest, coder_votes=votes)


def access(design, instance):
    audit = Path(f"audit/design-{design}")
    terminal = read(audit / "reconciled-terminal.json")
    follow = read(audit / "transport/protocol.json")["follow_up"]
    result = []
    window = read("audit/design-86/decision-windows.json")["windows"][str(instance)]
    if instance == 301:
        dispatch = read("audit/design-88/reconciled-terminal.json")
        dispatch_cases = {case["plan"]: case["witness"] for case in dispatch["cases"]}
    for case in terminal["cases"]:
        plan = case["plan"]
        relative = Path(f"probe/{instance}/{plan}")
        folder = audit / "physical-results" / relative
        for name in ("issued.jsonl", "completed.jsonl", "follow-up-actions.jsonl"):
            data = (ROOT / folder / name).read_bytes()
            recorded = terminal["source_snapshot"]["files"][str(relative / name)]
            assert (len(data), sha256(data).hexdigest()) == (recorded["bytes"], recorded["sha256"])
            if name != "follow-up-actions.jsonl":
                journal = data.splitlines(keepends=True)
                fork = window["after_action"]
                kind = name.removesuffix(".jsonl")
                route = window["routes"][plan]
                suffix_field = "action_suffix_sha256" if kind == "issued" else "observation_suffix_sha256"
                assert sha256(b"".join(journal[:fork])).hexdigest() == window["common_hashes"][kind]
                assert sha256(b"".join(journal[fork:])).hexdigest() == route[suffix_field]
                if instance == 301:
                    witness_file = f"probe/301/{dispatch_cases[plan]}/{name}"
                    assert recorded["sha256"] == dispatch["source_snapshot"]["files"][witness_file]["sha256"]
                    assert data == (ROOT / "audit/design-88/physical-results" / witness_file).read_bytes()
        stack = case["stack"]
        assert stack["passed"] and stack["original_terminal"] == dict(terminated=True, truncated=False, success=True)
        assert len(lines(folder / "issued.jsonl")) == len(lines(folder / "completed.jsonl")) == stack["completed"]
        final = json.loads(lines(folder / "completed.jsonl")[-1])
        poses = final["box_pose"]
        target0 = poses[follow["requested"]][0]
        other0 = poses[follow["nonrequested"]][0]
        rows = [json.loads(line) for line in lines(folder / "follow-up-actions.jsonl")]
        access = case["access"]
        assert len(rows) == access["actions"] <= follow["max_actions"]
        stable = 0
        for i, row in enumerate(rows, start=1):
            assert row["index"] == i and row["clock"][1] == final["clock"][1] + 4*i
            rise = row["target_position"][2] - target0[2]
            assert isclose(row["rise_m"], rise, abs_tol=1e-7)
            stable = stable + 1 if row["bounded"] and row["held_target"] and rise >= follow["rise_m"] else 0
            assert stable == row["stable"]
        displacement = max((dist(r["other_position"], other0) for r in rows), default=0.)
        assert isclose(displacement, access["peak_other_shift_m"], abs_tol=1e-15)
        rotation = max((r["other_rotation_rad"] for r in rows), default=0.)
        assert rotation == access["peak_other_rotation_rad"]
        assert displacement <= follow["other_translation_m"] and rotation <= follow["other_rotation_rad"]
        assert stable == access["stable_actions"]
        assert access["passed"] == (stable >= follow["stable_actions"])
        blind = read(audit / "blind-reconciliation.json")
        result.append(dict(plan=plan, stack_steps=stack["completed"], access_steps=len(rows),
                           access_passed=access["passed"], held_steps=stable,
                           start_base_xyz_m=final["base_pose"][0], start_target_xyz_m=target0,
                           start_other_xyz_m=other0, other_turn_rad=rotation,
                           peak_rise_m=max((r["rise_m"] for r in rows), default=0.),
                           other_shift_m=displacement, blind_pose_agreement=[blind["pose_matches"], blind["total"]],
                           masked_pose=masked_pose(audit, plan, follow, target0, other0, rows),
                           phases=Counter(r["phase"] for r in rows)))
    return result


if __name__ == "__main__":
    if sys.flags.optimize:
        raise SystemExit("Run with assertions enabled (omit -O).")
    for path, digest in read("sources.sha256.json").items():
        assert sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    print(json.dumps(dict(box_choices={n: choices(n) for n in (87, 90)}, shoe=shoe_choices(),
                          access={n: access(n, scene) for n, scene in ((93,314),(94,301),(95,301))}), indent=2))
