"""Train-only label protocol diagnostic against independently reviewed cases."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import random
from data_tools.labeling import VERDICT_LABEL_SYSTEM, derive_candidate_label, LABEL_PROTOCOL
from data_tools.teacher import TeacherClient, TeacherError, atomic_json, sha256, utc_now


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / 'local/train_v5_initial_review.jsonl'
    examples = [json.loads(line) for line in source.read_text().splitlines()]
    examples = [e for e in examples if e['id'].endswith(('00000', '00004'))]
    client = TeacherClient(root / 'local/teacher/train/verdict-diagnostic')
    phases = []
    for phase in ('original', 'permuted'):
        visible, mappings = [], {}
        for index, episode in enumerate(examples):
            entries = copy.deepcopy(episode['entries'])
            if phase == 'permuted':
                random.Random(517 + index).shuffle(entries)
            mapping = {}
            for position, entry in enumerate(entries):
                original = entry['id']
                if phase == 'permuted':
                    entry['id'] = f'x{position+1}'
                mapping[entry['id']] = original
            identifier = f'e{index+1}'
            mappings[identifier] = mapping
            visible.append({'id':identifier, 'context':episode['context'], 'entries':entries})
        if phase == 'permuted':
            visible.reverse()
        failures = []
        for attempt in range(3):
            user = {'episodes':visible}
            if attempt:
                user['format_reminder'] = 'Return valid JSON with no trailing commas or extra closing braces. Include all requested IDs.'
                user['format_retry'] = attempt
            try:
                result = client.complete_json(VERDICT_LABEL_SYSTEM, json.dumps(user,ensure_ascii=False), max_tokens=12288, phase='train-verdict-diagnostic-'+phase, request_id=f'{phase}-{attempt}')
                annotations = {item['id']:item for item in result.parsed['labels']}
                if set(annotations) != {item['id'] for item in visible}:
                    raise ValueError('Wrong diagnostic episode mapping')
                rows = []
                for index, episode in enumerate(examples):
                    identifier=f'e{index+1}'
                    item=next(v for v in visible if v['id']==identifier)
                    label=derive_candidate_label(annotations[identifier],item)
                    label['acceptable_ids']=[mappings[identifier][i] for i in label['acceptable_ids']]
                    required={'c7','c8'} if episode['id'].endswith('00000') else {'c8','c12'}
                    rows.append({'id':episode['id'],'label':label,'required_equivalents_found':required<=set(label['acceptable_ids']),'diagnostic': 'underspecified output-format exclusion' if episode['id'].endswith('00000') else 'equal patch stdout with unrequested exit-status difference'})
                phases.append({'phase':phase,'audit_id':result.audit_id,'usage':result.usage,'rows':rows,'prior_failures':failures})
                break
            except (TeacherError, ValueError) as exc:
                failures.append(str(exc))
        else:
            phases.append({'phase':phase,'failed':True,'failures':failures})
    report={'created_at':utc_now(),'protocol':LABEL_PROTOCOL,'source_sha256':sha256(source.read_bytes()),'source':'Train-only already rejected v5 diagnostic; never counted as new training data','phases':phases,'limits':'Two targeted engineering failures only; no general teacher-quality equivalence claim. All raw responses, including malformed JSON, retained. No answer hints sent to teacher.'}
    atomic_json(root/'data/train_verdict_diagnostic.report.json', report)
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
