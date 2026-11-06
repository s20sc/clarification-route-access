"""Offline test of a bounded clarification channel and equal-information decision rules."""
from collections import Counter
from fractions import Fraction
from itertools import product
import json
import random
from pathlib import Path

BOXES = ("box1", "box2")
BIND = {"box1": "box2_bottom", "box2": "box1_bottom"}
CONDITIONS = ("ordinary", "self_directed", "clarification_oriented")
SEED = 20261104059


def public_context(case):
    return {key: case[key] for key in ("visible_preference", "cue", "cue_reliability", "question_cost")}


def decision_rule(context, policy="regret"):
    visible = context["visible_preference"]
    if visible:
        return visible
    cue = context["cue"]
    risk = 1 - Fraction(context["cue_reliability"])
    if policy == "always_ask" or (policy == "regret" and risk > Fraction(context["question_cost"])):
        return "ask"
    assert policy in ("regret", "direct")
    return cue


class DecisionError(ValueError):
    pass


class Trial:
    """Environment owns the hidden follow-up requirement. Plans specify the TOP box."""

    def __init__(self, case):
        self.context = public_context(case)
        self._requirement = case["hidden_preference"]
        self.questions = 0
        self.calls = 0
        self.plan = None
        self.invalid = False

    def submit(self, action):
        if self.invalid or self.plan or self.calls >= 2:
            raise DecisionError("closed trial")
        self.calls += 1
        if action == "ask" and self.calls == 1 and self.context["visible_preference"] is None:
            self.questions = 1
            # This is the only delivery of the hidden requirement.
            return {"event": "user_reply", "top_requested": self._requirement}
        if not isinstance(action, str) or action not in BIND:
            self.invalid = True
            raise DecisionError("invalid action")
        hard = self._requirement if self.questions else self.context["visible_preference"]
        if hard and action != hard:
            self.invalid = True
            raise DecisionError("explicit request violated")
        self.plan = BIND[action]
        return {"event": "plan_bound", "plan": self.plan}

    def score(self, actual_top):
        if not self.plan or actual_top not in BIND or BIND[actual_top] != self.plan:
            raise DecisionError("physical endpoint absent or contradicts bound plan")
        correct = actual_top == self._requirement
        return dict(preference_met=correct, questions=self.questions, calls=self.calls,
                    normalized_loss=str((1-int(correct)) + self.questions * Fraction(self.context["question_cost"])))


def cases():
    # Within each cue/cost cell, stratify hidden demand and use exact proportions.
    # Cases are controlled design assignments, with no claim of observed human preferences.
    rows = []
    for reliability, cost, cue in product(("1/2", "3/4"), ("1/8", "3/8", "5/8"), BOXES):
        same = int(4 * Fraction(reliability))
        other, = set(BOXES) - {cue}
        for hidden in [cue] * same + [other] * (4 - same):
            rows.append(dict(visible_preference=None, cue=cue, cue_reliability=reliability,
                             question_cost=cost, hidden_preference=hidden))
    for cost, visible in product(("1/8", "3/8", "5/8"), BOXES):
        rows.append(dict(visible_preference=visible, cue=None, cue_reliability=None,
                         question_cost=cost, hidden_preference=visible))
    random.Random(SEED).shuffle(rows)
    return rows


def simulate_rule(case, policy):
    trial = Trial(case)
    decision = decision_rule(trial.context, policy)
    first = trial.submit(decision)
    if decision == "ask":
        decision = first["top_requested"]
        assert trial.submit(decision)["plan"] == BIND[decision]
    return trial.score(decision)


def reject(fn):
    try:
        fn()
    except DecisionError:
        return 1
    raise AssertionError("negative control was accepted")


def run():
    rows = cases()
    assert len(rows) == 54 and len({json.dumps(public_context(r), sort_keys=True) for r in rows}) == 18
    assert all(set(public_context(r)) == {"visible_preference", "cue", "cue_reliability", "question_cost"} for r in rows)
    secret = next(r for r in rows if r["visible_preference"] is None and r["hidden_preference"] != r["cue"])
    same = dict(secret, hidden_preference=secret["cue"])
    assert public_context(secret) == public_context(same)
    assert Trial(secret).submit(secret["cue"]) == Trial(same).submit(same["cue"])
    fixed = next(r for r in rows if r["visible_preference"] == "box1")
    bad = reject(lambda: Trial(fixed).submit("ask")) + reject(lambda: Trial(fixed).submit("box2"))
    bad += reject(lambda: Trial(secret).submit("reset")) + reject(lambda: Trial(secret).score("box1"))
    again = Trial(secret)
    assert again.submit("ask") == {"event": "user_reply", "top_requested": secret["hidden_preference"]}
    bad += reject(lambda: again.submit("ask"))
    contradiction = Trial(secret)
    contradiction.submit("box1")
    bad += reject(lambda: contradiction.score("box2"))
    response = Trial(secret)
    reply = response.submit("ask")["top_requested"]
    wrong, = set(BOXES) - {reply}
    bad += reject(lambda: response.submit(wrong))
    invalid = Trial(secret)
    bad += reject(lambda: invalid.submit([]))
    bad += reject(lambda: invalid.submit(secret["cue"]))
    counts = {}
    for policy in ("direct", "always_ask", "regret"):
        results = [simulate_rule(row, policy) for row in rows]
        counts[policy] = dict(preference_met=sum(row["preference_met"] for row in results),
                              questions=sum(row["questions"] for row in results),
                              calls=sum(row["calls"] for row in results),
                              total_normalized_loss=str(sum(Fraction(row["normalized_loss"]) for row in results)))
    for r, c in product(("1/2", "3/4"), ("1/8", "3/8", "5/8")):
        block = [row for row in rows if row["cue_reliability"] == r and row["question_cost"] == c]
        assert len(block) == 8
        assert Counter(row["cue"] for row in block) == Counter(dict(box1=4, box2=4))
        assert sum(row["hidden_preference"] == row["cue"] for row in block) == 8*Fraction(r)
    regret_bound = sum(8 * min(1-Fraction(r), Fraction(c))
                       for r,c in product(("1/2","3/4"),("1/8","3/8","5/8")))
    assert Fraction(counts["regret"]["total_normalized_loss"]) == regret_bound == 13
    root = Path(__file__).resolve().parents[2]
    audit = root / "audit/design-59"
    receipt = dict(registered_controlled_assignments=54, distinct_public_contexts=18, rejected_negative_controls=bad,
                   oracle_rule_receipts=counts, model_calls=0, physical_dispatches=0,
                   scope="CPU binding and analytically optimal risk-neutral rule under the synthetic design. Outcomes shown here are rule simulation, not human observations or robot execution.")
    (audit/"retrieval-contract-receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    (audit/"retrieval-private-assignments.json").write_text(json.dumps(dict(seed=SEED, cases=rows),indent=2)+"\n")
    print(json.dumps(receipt,indent=2))


if __name__ == "__main__":
    run()
