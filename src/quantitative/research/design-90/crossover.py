"""Freeze a counterfactual cost crossover, reusing the verified D87 model runner."""
import importlib.util
import itertools
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "audit/design-90"
RUNNER = ROOT / "research/design-87/run_choice.py"
RECONCILER = ROOT / "research/design-87/reconcile_choice.py"
SEED = 20261104090


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pilot = load("choice_pilot", RUNNER)


def cases():
    for gap, cue in itertools.product((86, 1150), ("box2", "box1")):
        other, = set(("box1", "box2")) - {cue}
        yield gap, dict(visible_preference=None, cue=cue, cue_reliability="3/4",
                        question_cost="1/8", hidden_preference=other)
    for box in ("box1", "box2"):
        yield 86, dict(visible_preference=box, cue=None, cue_reliability=None,
                       question_cost="1/8", hidden_preference=box)


def prepare():
    old = pilot.read(ROOT / "audit/design-87/matrix.json")
    intake = pilot.read(AUDIT / "backend-intake.json")
    assert intake["model"] == old["model"]["model"] and intake["digest"] == old["model"]["model_digest"]
    windows = pilot.read(pilot.WINDOWS)["windows"]
    inputs = list(cases())
    matrix = []
    for instance, i, swapped, reverse in itertools.product(windows, range(len(inputs)), (False, True), (False, True)):
        gap, private = inputs[i]
        window = windows[instance]
        labels = ("box2", "box1") if swapped else ("box1", "box2")
        mapping = dict(zip(pilot.ALIASES, labels))
        order = pilot.ALIASES[::-1] if reverse else pilot.ALIASES
        context = pilot.BRIDGE["CONTRACT"]["public_context"](private)
        packet = pilot.BRIDGE["public_observation"](window, mapping, order, context)
        for plan in packet["scene"]["plans"]:
            plan["excess_motor_steps"] = gap if mapping[plan["upper"]] == "box2" else 0
        costs = pilot.expected_cost(packet)
        baseline = min(costs, key=lambda key: (costs[key], key))
        matrix.append(dict(id=f"{instance}-{i}-{int(swapped)}{int(reverse)}", instance=instance, index=i,
                           hypothetical_cost_gap=gap, map=mapping, private=private, packet=packet, baseline=baseline,
                           losses={label: str(cost) for label, cost in costs.items()}))
    assert len(matrix) == 48
    design = {key: old[key] for key in ("model", "profiles", "format", "options", "max_first_calls", "max_total_calls")}
    design["model"]["ollama_version"] = intake["observed_version"]
    design.update(cases=matrix, seed=SEED,
                  prompt=old["prompt"] + " The motor-step estimates are hypothetical in this decision test. Use the stated estimates.",
                  primary="Final canonical route, valid and invariant over all four alias/order presentations.",
                  support="Compare profiles in each scene and hypothetical cost condition. Cross-scene robustness requires a consistent contrast under the same estimated cost and cue.",
                  limits="Exploratory follow-up after D87. One model, two frozen observations, two cost schedules and two deliberately misleading cue/reply pairs. No physical dispatch or population utility inference.",
                  source_hashes={str(p.relative_to(ROOT)): pilot.sha(p.read_bytes()) for p in (
                      RUNNER, RECONCILER, Path(__file__).resolve(), pilot.WINDOWS, pilot.BRIDGE_FILE,
                      pilot.HELPER_FILE, pilot.MODEL_FILE, ROOT / "audit/design-87/matrix.json", AUDIT / "backend-intake.json")})
    path = AUDIT / "matrix.json"
    assert not path.exists()
    pilot.write(path, design)


def execute():
    pilot.AUDIT = AUDIT
    pilot.SEED = SEED
    pilot.execute()


def summarize():
    rec = load("choice_reconciler", RECONCILER)
    rec.AUDIT = AUDIT
    rec.main()


if __name__ == "__main__":
    {"prepare": prepare, "execute": execute, "summarize": summarize}[sys.argv[1]]()
