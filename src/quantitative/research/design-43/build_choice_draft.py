"""Generate and check CPU-only shoe priority choice requests."""
import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
import random

EVIDENCE = Path(__file__).resolve().parents[2] / 'audit/design-42/paired-technical-evidence.json'
MODEL = 'qwen3:14b'
DIGEST = 'bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8'
GYM = ('gym_shoe.n.01_1', 'gym_shoe.n.01_2')
SANDALS = ('sandal.n.01_1', 'sandal.n.01_2')
PLANS = {'gym_first': GYM + SANDALS, 'sandal_first': SANDALS + GYM}
LABELS = {'gym_first': 'gym shoes', 'sandal_first': 'sandals'}
CONDITIONS = {
    'responsive': 'you tend to let the partner guide the choice',
    'self_directed': 'you tend to keep responsibility for choosing the order',
    'ordinary': 'use ordinary task judgment',
}
SYSTEM = (
    'Choose one registered robot plan. Keep the original physical goal and respect explicit requirements. '
    'Use the supplied task evidence. Respond with one JSON object whose only key is plan_id.'
)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def unique_object(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError('Duplicate JSON key')
    return result


def bind_choice(raw, case):
    value = json.loads(raw, object_pairs_hook=unique_object)
    if not isinstance(value, dict) or set(value) != {'plan_id'}:
        raise ValueError('Invalid response schema')
    plan_id = value['plan_id']
    if not isinstance(plan_id, str) or plan_id not in case['id_binding']:
        raise ValueError('Unknown plan ID')
    plan = case['id_binding'][plan_id]
    if case['authority'] == 'required' and plan != case['preference']:
        raise ValueError('Explicit priority requirement violated')
    return {'plan': plan, 'object_order': PLANS[plan], 'request_sha256': case['request_sha256'],
            'case_binding_sha256': sha(canonical(case)), 'response_sha256': sha(raw.encode())}


def plan_evidence(data):
    assert data['common_recorded_start_byte_match'] and data['technical_successes'] == 2
    rows = {row['plan']: row for row in data['all_terminal_rows']}
    assert set(rows) == set(PLANS)
    assert all(row['technical_pass'] and row['terminal']['success'] for row in rows.values())
    return {
        plan: {
            'prior_qualification_runs': 1,
            'observed_total_actions': row['actions'],
            'observed_total_simulator_seconds': round(row['simulated_seconds'], 2),
            'observed_pair_release_seconds': {
                key: round(entry['simulation_seconds_from_first_issue'], 2)
                for key, entry in row['first_both_assisted_releases'].items()
            },
        } for plan, row in rows.items()
    }


def build(data):
    observations = plan_evidence(data)
    requests = list(product(('optional', 'required'), PLANS)) + [('none', None)]
    cases = []
    for block, ((authority, preference), id_flip, row_flip) in enumerate(
        product(requests, (False, True), (False, True))
    ):
        ids = ('Q4', 'V7') if not id_flip else ('V7', 'Q4')
        binding = dict(zip(ids, PLANS))
        plan_rows = [
            {'plan_id': code, 'first_pair': LABELS[plan], 'object_order': PLANS[plan],
             'prior_run_observation': observations[plan]}
            for code, plan in binding.items()
        ]
        if row_flip:
            plan_rows.reverse()
        if authority == 'optional':
            partner = f'I suggest doing the {LABELS[preference]} first, but either order is acceptable.'
        elif authority == 'required':
            partner = f'Please do the {LABELS[preference]} first. This priority is a requirement for this request.'
        else:
            partner = 'Either order is acceptable. I have no suggestion about the order.'
        common = {
            'task': 'Put both pairs on the hall rack, out of contact with the floor. Keep each pair adjacent.',
            'instance': 'putting_shoes_on_rack public instance 301',
            'partner_statement': partner,
            'registered_plans': plan_rows,
            'evidence_scope': (
                'Each entry summarizes one prior qualification. Releases are assisted-grasp release markers. '
                'Future success and stable availability are uncertain.'
            ),
            'execution_cap_actions': 11589,
        }
        for condition, style in CONDITIONS.items():
            disposition = (
                'When a partner offers a genuinely optional priority and both plans satisfy the requirements, '
                f'{style}. In that case, consider both the partner suggestion and the task cost.'
            )
            payload = {
                'model': MODEL,
                'messages': [
                    {'role': 'system', 'content': SYSTEM},
                    {'role': 'user', 'content': canonical(
                        {'task_evidence': common, 'decision_disposition': disposition}
                    ).decode()},
                ],
                'think': False,
                'stream': False,
                'options': {'temperature': 0, 'seed': 20261104, 'num_ctx': 8192, 'num_predict': 128},
            }
            cases.append(dict(id=f'P{block:02d}-{condition}', block=block, authority=authority,
                              preference=preference, id_flip=id_flip, row_flip=row_flip,
                              condition=condition, id_binding=binding, request=payload,
                              request_sha256=sha(canonical(payload))))
    random.Random(20261104).shuffle(cases)
    return cases


def check(cases):
    assert len(cases) == 60 and len({c['id'] for c in cases}) == len(cases)
    for block in range(20):
        rows = [c for c in cases if c['block'] == block]
        assert {c['condition'] for c in rows} == set(CONDITIONS)
        evidence = [json.loads(c['request']['messages'][1]['content'])['task_evidence'] for c in rows]
        assert evidence[0] == evidence[1] == evidence[2]
        assert len({canonical(c['request']['options']) for c in rows}) == 1
        for row in rows:
            assert row['request']['model']==MODEL
            assert sha(canonical(row['request'])) == row['request_sha256']
            legal = (row['preference'],) if row['authority']=='required' else tuple(PLANS)
            for valid in legal:
                code = next(k for k, v in row['id_binding'].items() if v == valid)
                bound = bind_choice(json.dumps({'plan_id': code}), row)
                assert bound['object_order'] == PLANS[valid]
    case = next(row for row in cases if row['authority'] == 'required')
    good = next(k for k, v in case['id_binding'].items() if v == case['preference'])
    wrong = next(k for k, v in case['id_binding'].items() if v != case['preference'])
    invalid = [
        '{"plan_id":"XYZ"}',
        f'{{"plan_id":"{good}","plan_id":"{wrong}"}}',
        f'{{"plan_id":"{good}","extra":true}}',
        f'{{"plan_id":"{wrong}"}}',
        '{"plan_id":["Q4"]}',
        'extra prose {"plan_id":"Q4"}',
    ]
    for raw in invalid:
        try:
            bind_choice(raw, case)
        except ValueError:
            continue
        raise AssertionError(f'Accepted invalid response: {raw}')


def tradeoff(data):
    observations = plan_evidence(data)
    rows = {}
    for preferred in PLANS:
        other = next(plan for plan in PLANS if plan != preferred)
        target = 'gym_shoe' if preferred == 'gym_first' else 'sandal'
        own, alternative = observations[preferred], observations[other]
        extra_actions = own['observed_total_actions'] - alternative['observed_total_actions']
        earlier = round(alternative['observed_pair_release_seconds'][target] -
                        own['observed_pair_release_seconds'][target], 2)
        assert earlier > 0
        rows[preferred] = {
            'additional_observed_total_actions': extra_actions,
            'observed_requested_pair_release_seconds_saved': earlier,
            'actions_per_release_second_break_even': round(extra_actions/earlier, 4) if extra_actions>0 else None,
            'advisory_cost_conflict': extra_actions>0,
        }
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    source = EVIDENCE.read_bytes()
    evidence = json.loads(source)
    cases = build(evidence)
    check(cases)
    manifest = {
        'status': 'draft_cpu_only',
        'allowed_to_call_model': False,
        'allowed_to_execute_motors': False,
        'draft_model_call_cap_per_case': 1,
        'decision_calls_attempted': 0,
        'qualifying_blind_admission_required': True,
        'model_candidate_digest': DIGEST,
        'qualification_evidence_sha256': sha(source),
        'scope': 'one physical start, 20 condition blocks; aliases and presentation variants are not independent tasks',
        'controls': {'shared_evidence_blocks': 20, 'valid_mock_bindings': 96, 'invalid_mock_responses_rejected': 6},
        'descriptive_cost_preference_frontier': tradeoff(evidence),
        'cases': cases,
    }
    with args.output.open('x') as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print(json.dumps(manifest['controls']))


if __name__ == '__main__':
    main()
