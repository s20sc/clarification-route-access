"""Run the fixed shoe-order choice diagnostic after source and visual admission."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
import time
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
VERSION = HERE.parents[1]
BASE = 'http://127.0.0.1:11434'
MATRIX = VERSION / 'audit/design-46/choice-matrix.json'
BINDER = VERSION / 'research/design-43/build_choice_draft.py'
EVIDENCE = VERSION / 'audit/design-46/qualified-320.json'
A45 = VERSION / 'audit/design-45'
CLIPS = A45 / 'independent-review-320'
ADMISSION = A45 / '320-blind-admission.json'
RULES = VERSION / 'audit/design-46/fixed-rule-predictions.json'
CONTRACT = HERE / 'choice-contract.json'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, obj):
    with path.open('x') as stream:
        json.dump(obj, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def verify_sources():
    if sys.flags.optimize:
        raise RuntimeError('Source and admission assertions must be enabled')
    freeze=read(HERE/'input-freeze.json')
    for name,want in freeze['files'].items():
        if digest(HERE/name)!=want:
            raise ValueError(f'Frozen pilot input changed: {name}')
    contract = read(CONTRACT)
    for key, path in [
        ('matrix', MATRIX), ('binder', BINDER), ('evidence', EVIDENCE), ('rules', RULES),
        ('physical_archive', A45 / 'physical-results/latest-reconciled.json'),
        ('D44_comparison', VERSION / 'audit/design-44/decision-analysis.json'),
    ]:
        if contract['sha256'][key] != digest(path):
            raise ValueError(f'{key} source hash changed')
    matrix = read(MATRIX)
    assert matrix['status'] == 'draft_cpu_only' and matrix['decision_calls_attempted'] == 0
    assert matrix['qualification_evidence_sha256'] == contract['sha256']['evidence']
    cases = matrix['cases']
    assert len(cases) == 60 and len({r['id'] for r in cases}) == 60
    assert contract['instance'] == 320 and not contract['physical_followthrough_case_ids']
    conflicts = {row['block'] for row in cases if
                 (row['authority'],row['preference']) == ('optional','sandal_first')}
    assert sorted(conflicts) == contract['primary_conflict_blocks']
    return contract, matrix


def verify_admission():
    gate = read(ADMISSION)
    assert gate['shoe_pair_visual_order_admitted'] is True
    assert gate['instance'] == 320
    assert gate['qualified_evidence_sha256'] == digest(EVIDENCE)
    assert gate['physical_archive_sha256'] == digest(A45 / 'physical-results/latest-reconciled.json')
    assert set(gate['report_sha256']) == {'coder-1-report.json', 'coder-2-report.json'}
    for name, expected in gate['report_sha256'].items():
        assert digest(A45/name) == expected
    assert set(gate['clips']) == {'clip-01','clip-02'}
    assert {row['first_pair'] for row in gate['clips'].values()} == {'gym_shoe','sandal'}
    for clip, row in gate['clips'].items():
        assert digest(CLIPS / clip / 'video.mp4') == row['video_sha256']
    return gate


def api(path, data=None, timeout=4):
    request = Request(BASE + path, data=data, headers={'Content-Type':'application/json'})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def verify_model(contract):
    version = api('/api/version')['version']
    candidates = [m for m in api('/api/tags')['models'] if m['name']==contract['model']]
    assert version == contract['ollama_version']
    assert len(candidates)==1 and candidates[0]['digest']==contract['model_digest']
    return {'version':version, 'name':candidates[0]['name'], 'digest':candidates[0]['digest']}


def result_binding(response, case, bind):
    if response.get('model') != case['request']['model']:
        raise ValueError('Model name mismatch')
    if response.get('done') is not True or response.get('done_reason') != 'stop':
        raise ValueError('Incomplete or length-limited response')
    output_tokens=response.get('eval_count')
    if type(output_tokens) is not int or not 0 < output_tokens <= case['request']['options']['num_predict']:
        raise ValueError('Missing or invalid output token receipt')
    return bind(response['message']['content'],case)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path)
    parser.add_argument('--check-sources',action='store_true')
    args=parser.parse_args()
    contract,matrix=verify_sources()
    if args.check_sources:
        assert args.output is None
        print('Frozen sources and primary conflict blocks verified. No HTTP inference.')
        return
    assert args.output is not None
    admission=verify_admission()
    model=verify_model(contract)
    helpers=runpy.run_path(str(BINDER))
    canonical=helpers['canonical']
    output=args.output
    output.mkdir(exist_ok=False)
    dump(output/'invocation.json',dict(contract_sha256=digest(CONTRACT),admission_sha256=digest(ADMISSION),
         matrix_sha256=digest(MATRIX),model=model,order=[c['id'] for c in matrix['cases']],
         physical_followthrough_case_ids=contract['physical_followthrough_case_ids'],
         visual_admission=admission, call_cap=1, network_timeout_seconds=contract['network_timeout_seconds']))
    records=[]
    for case in matrix['cases']:
        stem=output/case['id']
        request=canonical(case['request'])
        assert hashlib.sha256(request).hexdigest()==case['request_sha256']
        stem.with_suffix('.request.json').write_bytes(request)
        record={'case_id':case['id'],'block':case['block'],'authority':case['authority'],'condition':case['condition'],
                'preference':case['preference'],'call_issued':True,'response_received':False,'binding':None}
        dump(stem.with_suffix('.issued.json'),dict(case_id=case['id'],request_sha256=case['request_sha256'],issued_unix=time.time()))
        started=time.monotonic()
        try:
            response=api('/api/chat',request,contract['network_timeout_seconds'])
            record['response_received']=True
            dump(stem.with_suffix('.response.json'),response)
            record['binding']=result_binding(response,case,helpers['bind_choice'])
        except Exception as exc:
            record['failure']={'type':type(exc).__name__,'message':str(exc)}
        record['elapsed_wall_seconds']=time.monotonic()-started
        dump(stem.with_suffix('.result.json'),record)
        records.append(record)
    try:
        end_model=verify_model(contract)
    except Exception:
        end_model=None
    dump(output/'complete.json',dict(records=records,model_end_matches=(end_model==model),action_runs=0,
        invalid_cases=sum(r['binding'] is None for r in records),
        calls_issued=len(records), responses_received=sum(r['response_received'] for r in records)))


if __name__=='__main__':
    main()
