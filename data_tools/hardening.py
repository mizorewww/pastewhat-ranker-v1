"""Autonomous new-Train mining, blind re-review and immutable hardening mixture.

Only a Dev-selected v0 may nominate new Train inputs. Predictions stay in the
mining sidecar and never enter teacher requests or student features. Original
teacher labels are preserved; any post-mining disagreement rejects the episode.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import fcntl
import json
from pathlib import Path
import random
import subprocess
import sys
import time

from data_tools.content import content_fingerprint
from data_tools.deployment import placement_issue
from data_tools.difficulty import describe_difficulty
from data_tools.freeze import choose_registered, publish_bytes
from data_tools.generate import CACHE_VERSION, PARTITION_PATH, PROMPT_VERSION, ROOT
from data_tools.labeling import VERDICT_LABEL_SYSTEM, derive_candidate_label, LABEL_PROTOCOL
from data_tools.rate_limit import AccountCoordinator
from data_tools.replay import ReplayVerifier
from data_tools.teacher import TeacherClient, TeacherError, atomic_json, canonical_bytes, sha256, utc_now
from pastewhat_ranker.preprocess import Preprocessor
from run_contract import load_run_plan


def read_train(path, plan):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT) or any(value in str(path.relative_to(ROOT)).lower() for value in ('calibration', 'test', 'heldout')):
        raise ValueError('Hardening may read only owned Train artifacts')
    allowed = {family['id'] for family in json.loads(PARTITION_PATH.read_text())['families']['train']}
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        if row['family_id'] not in allowed or any(row.get('provenance', {}).get(key) != value for key, value in plan.binding().items()):
            raise ValueError('A hardening row crosses family or run ownership')
    return rows


def same_action(first, second):
    same = first['decision'] == second['decision'] and set(first['acceptable_ids']) == set(second['acceptable_ids'])
    reasons = {first['abstain_reason'], second['abstain_reason']}
    return same and (len(reasons) == 1 or reasons == {'ambiguous', 'insufficient_context'})


def request_review_labels(visible, client, key, pass_index):
    failures = []
    for retry in range(3):
        body = {'episodes': visible}
        if retry:
            body['format_retry'] = retry
            body['format_requirement'] = 'Return valid JSON, every episode and every candidate ID exactly once, booleans and exact positive-text quotations. Do not use any previous answers.'
        try:
            result = client.complete_json(VERDICT_LABEL_SYSTEM, json.dumps(body, ensure_ascii=False), max_tokens=16384, response_format='json_object', phase=f'train-post-mining-blind-{pass_index}', request_id=key + f'-{pass_index}-{retry}')
            annotations = result.parsed.get('labels', [])
            if len(annotations) != len(visible) or {item.get('id') for item in annotations} != {row['id'] for row in visible}:
                raise ValueError('Post-mining reviewer omitted or remapped episodes')
            inputs = {row['id']: row for row in visible}
            derived = {item['id']: derive_candidate_label(item, inputs[item['id']]) for item in annotations}
            return result, derived, failures
        except TeacherError as exc:
            if any(term in str(exc) for term in ('HTTP', 'transport', 'quota', 'account paused')):
                raise
            failures.append({'attempt': retry, 'error': str(exc)})
        except (ValueError, KeyError, TypeError) as exc:
            failures.append({'attempt': retry, 'error': str(exc), 'audit_id': result.audit_id})
    return None, None, failures


def blind_review(group, client, destination, plan):
    key = sha256(canonical_bytes({'ids': [row['id'] for row in group], 'content': [content_fingerprint(row) for row in group], 'plan': plan.binding(), 'prompt': VERDICT_LABEL_SYSTEM}))
    path = destination / f'{key}.json'
    if path.is_file():
        return json.loads(path.read_text())
    observed, audit_ids = {}, []
    for pass_index in (0, 1):
        visible, mappings = [], {}
        for index, episode in enumerate(group):
            entries = copy.deepcopy(episode['entries'])
            random.Random(int(content_fingerprint(episode)[:16], 16) ^ (71923 + pass_index)).shuffle(entries)
            opaque = f'hr{pass_index}-{index+1}'
            mapping = {}
            for position, entry in enumerate(entries):
                identifier = f'q{pass_index}-{position+1}'
                mapping[identifier] = entry['id']
                entry['id'] = identifier
            mappings[opaque] = (episode, mapping)
            visible.append({'id': opaque, 'context': episode['context'], 'entries': entries})
        if pass_index:
            visible.reverse()
        result, labels, failures = request_review_labels(visible, client, key, pass_index)
        if result is None:
            record = {**plan.binding(), 'label_protocol': LABEL_PROTOCOL, 'accepted': [], 'rejected': [{'id': row['id'], 'original_label': row['label'], 'reason': 'Three independent response-format attempts failed; no label edited', 'format_failures': failures} for row in group], 'reviewed_at': utc_now(), 'student_predictions_in_teacher_input': False, 'teacher_label_edits': 0}
            atomic_json(path, record)
            return record
        for identifier, derived in labels.items():
            original, mapping = mappings[identifier]
            derived['acceptable_ids'] = [mapping[value] for value in derived['acceptable_ids']]
            observed.setdefault(original['id'], []).append(derived)
        audit_ids.append(result.audit_id)
    accepted, rejected = [], []
    for episode in group:
        labels = observed[episode['id']]
        issue = placement_issue(episode)
        if issue or not all(same_action(episode['label'], label) for label in labels) or not same_action(labels[0], labels[1]):
            rejected.append({'id': episode['id'], 'original_label': episode['label'], 'review_labels': labels, 'reason': issue or 'post-mining blind teachers do not all preserve the original action set', 'audit_ids': audit_ids})
            continue
        accepted.append({'id': episode['id'], 'content_sha256': content_fingerprint(episode), 'original_label': episode['label'], 'review_labels': labels, 'audit_ids': audit_ids, 'reason_agreement': len({episode['label']['abstain_reason'], labels[0]['abstain_reason'], labels[1]['abstain_reason']}) == 1})
    record = {**plan.binding(), 'label_protocol': LABEL_PROTOCOL, 'accepted': accepted, 'rejected': rejected, 'reviewed_at': utc_now(), 'student_predictions_in_teacher_input': False, 'teacher_label_edits': 0}
    atomic_json(path, record)
    return record


def ensure_idle_v0(plan):
    directory = ROOT / plan.pipeline_directory
    ready_path, status_path = directory / 'ranker-v0-ready.json', directory / 'status.json'
    if not ready_path.is_file() or not status_path.is_file():
        return None
    ready, status = json.loads(ready_path.read_text()), json.loads(status_path.read_text())
    if any(ready.get(key) != value or status.get(key) != value for key, value in plan.binding().items()):
        raise ValueError('v0 handoff has a different registered plan')
    if ready.get('selected_by') != 'Dev only':
        raise ValueError('v0 was not selected only on Dev')
    if status.get('phase') != 'hardening_preparation' or status.get('status') != 'waiting_for_frozen_train_dev_data':
        return None
    expected = ROOT / plan.data_path('hardening')
    hard = plan.document['hardening']
    if expected.exists() or Path(status.get('required_path', '')).resolve() != expected.resolve() or status.get('required_count') != hard['accepted_new'] + hard['retained_original']:
        return None
    return ready


def run_command(command, log, plan):
    plan.verify_unchanged()
    with log.open('a') as stream:
        stream.write(json.dumps({'started_at': utc_now(), 'command': command}) + '\n')
        stream.flush()
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Hardening phase failed with status {result.returncode}; see {log.relative_to(ROOT)}')


def verify_post_mining_review(episode, record, replay):
    """Replay both new blind request views before publishing cached approval."""
    if record['original_label'] != episode['label'] or record['content_sha256'] != content_fingerprint(episode) or len(set(record['audit_ids'])) != 2:
        raise ValueError('Post-mining approval is not bound to this unchanged episode')
    for pass_index, audit_id in enumerate(record['audit_ids']):
        audit, response = replay.audit(audit_id, 'hardcase')
        request = audit['request']
        if request['messages'][0]['content'] != VERDICT_LABEL_SYSTEM or audit['phase'] != f'train-post-mining-blind-{pass_index}':
            raise ValueError('Post-mining label protocol or independent pass differs')
        user = json.loads(request['messages'][1]['content'])
        if set(user) - {'episodes', 'format_retry', 'format_requirement'} or any(set(row) != {'id', 'context', 'entries'} for row in user['episodes']):
            raise ValueError('Post-mining teacher saw fields beyond deployment input')
        visible = replay.visible_match(audit, episode, ignore_order=True)
        entries = copy.deepcopy(episode['entries'])
        random.Random(int(content_fingerprint(episode)[:16], 16) ^ (71923 + pass_index)).shuffle(entries)
        mapping = {f'q{pass_index}-{index+1}': entry['id'] for index, entry in enumerate(entries)}
        for index, entry in enumerate(entries):
            entry['id'] = f'q{pass_index}-{index+1}'
        if entries != visible['entries']:
            raise ValueError('Post-mining candidate permutation does not replay')
        annotation = next(row for row in response['labels'] if row['id'] == visible['id'])
        observed = derive_candidate_label(annotation, visible)
        observed['acceptable_ids'] = [mapping[value] for value in observed['acceptable_ids']]
        if observed != record['review_labels'][pass_index] or not same_action(observed, episode['label']):
            raise ValueError('Post-mining accepted action does not replay to both teacher responses')


def publish_mixture(original, pool, reviews, selection, plan, output_dir, preprocessor):
    hard = plan.document['hardening']
    by_id = {row['id']: row for row in pool}
    approved = {row['id']: row for record in reviews for row in record['accepted']}
    priority_ids = [row['id'] for row in selection['selected']]
    chosen_ids = [identifier for identifier in priority_ids if identifier in approved][:hard['accepted_new']]
    if len(chosen_ids) < hard['accepted_new']:
        atomic_json(output_dir / 'insufficient-reviewed-new-data.json', {**plan.binding(), 'required': hard['accepted_new'], 'accepted': len(chosen_ids), 'rejected': sum(len(record['rejected']) for record in reviews), 'status': 'requires_new_train_pool_not_label_repair'})
        raise ValueError('Too few independently confirmed new examples; do not repeat reviewers until a preferred label appears')
    old_contents = {content_fingerprint(row) for row in original}
    new = []
    replay = ReplayVerifier(ROOT, 'train', preprocessor)
    for identifier in chosen_ids:
        row = copy.deepcopy(by_id[identifier])
        replay.verify(row)
        verify_post_mining_review(row, approved[identifier], replay)
        fingerprint = content_fingerprint(row)
        if fingerprint in old_contents or placement_issue(row):
            raise ValueError('Reviewed new hardening example duplicates an original or is locally rejected')
        row['provenance']['post_mining_review'] = approved[identifier]
        new.append(row)
    retained = choose_registered(original, json.loads(PARTITION_PATH.read_text()), 'train', hard['retained_original'])
    chosen = retained + new
    random.Random(42).shuffle(chosen)
    destination = ROOT / plan.data_path('hardening')
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = b''.join(canonical_bytes(row) + b'\n' for row in chosen)
    if destination.exists() and destination.read_bytes() != payload:
        raise ValueError('Immutable hardening snapshot already exists with different contents')
    fingerprints = [{'id': row['id'], 'family_id': row['family_id'], 'content_sha256': content_fingerprint(row)} for row in chosen]
    if len({row['content_sha256'] for row in fingerprints}) != len(chosen):
        raise ValueError('Hardening mixture repeats visible content')
    fingerprint_path = destination.with_suffix('.fingerprints.jsonl')
    publish_bytes(fingerprint_path, b''.join(canonical_bytes(row) + b'\n' for row in fingerprints))
    manifest = {**plan.binding(), 'stage': 'hardening', 'split': 'train', 'episodes': len(chosen), 'sha256': sha256(payload), 'family_partition_sha256': sha256(PARTITION_PATH.read_bytes()), 'created_at': utc_now(), 'path': str(destination.relative_to(ROOT)), 'retained_original': len(retained), 'accepted_new': len(new), 'original_rows_preserved_byte_for_byte': True, 'new_content_disjoint_from_original': True, 'labels': dict(Counter(row['label']['decision'] if row['label']['decision'] == 'select' else row['label']['abstain_reason'] for row in chosen)), 'families': dict(Counter(row['family_id'] for row in chosen)), 'preprocessing': preprocessor.manifest(), 'fingerprints': str(fingerprint_path.relative_to(ROOT)), 'mining_proposals_sha256': selection['proposals_sha256'], 'post_mining_review_accepted': len(approved), 'post_mining_review_rejected': sum(len(record['rejected']) for record in reviews), 'student_disagreement_is_not_teacher_truth': True, 'human_validated': False, 'teacher_label_edits': 0}
    manifest['difficulty_profile'] = describe_difficulty(chosen)
    plan.verify_unchanged()
    publish_bytes(destination.with_suffix('.manifest.json'), json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2).encode() + b'\n')
    publish_bytes(destination, payload)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-plan', required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=5)
    parser.add_argument('--poll-seconds', type=float, default=30)
    args = parser.parse_args()
    plan = load_run_plan(args.run_plan)
    if plan.document['teacher_contract_version'] != PROMPT_VERSION or not 1 <= args.workers <= 4 or not 1 <= args.batch_size <= 10:
        raise SystemExit('Invalid hardening production contract')
    directory = ROOT / 'local/hardening-production' / plan.run_id
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / '.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    completed_snapshot = ROOT / plan.data_path('hardening')
    if completed_snapshot.is_file():
        completed = json.loads(completed_snapshot.with_suffix('.manifest.json').read_text())
        required = plan.document['hardening']['accepted_new'] + plan.document['hardening']['retained_original']
        if any(completed.get(key) != value for key, value in plan.binding().items()) or completed.get('episodes') != required or completed.get('sha256') != sha256(completed_snapshot.read_bytes()):
            raise ValueError('Existing hardening snapshot does not match this registered run')
        atomic_json(directory / 'complete.json', completed)
        print(json.dumps({'status': 'already_frozen', **plan.binding()}))
        return
    coordinator = AccountCoordinator()
    while True:
        plan.verify_unchanged()
        ready = ensure_idle_v0(plan)
        if ready:
            break
        atomic_json(directory / 'status.json', {**plan.binding(), 'phase': 'waiting_for_dev_selected_v0', 'updated_at': utc_now()})
        time.sleep(min(60, max(5, args.poll_seconds)))
    original_path = ROOT / plan.data_path('train')
    original = read_train(original_path, plan)
    ready_path = ROOT / plan.pipeline_directory / 'ranker-v0-ready.json'
    pool_path = ROOT / 'local/data-production' / plan.run_id / 'train_hard-pool.jsonl'
    desired = plan.document['hardening']['pool_episodes']
    while not pool_path.is_file() or len(pool_path.read_text().splitlines()) != desired:
        state = coordinator.status()
        if state['paused']:
            atomic_json(directory / 'status.json', {**plan.binding(), 'phase': 'waiting_for_account_quota', 'account_rate_state': state, 'updated_at': utc_now()})
            time.sleep(min(60, max(5, args.poll_seconds)))
            continue
        atomic_json(directory / 'status.json', {**plan.binding(), 'phase': 'new_train_pool', 'updated_at': utc_now()})
        try:
            run_command([sys.executable, '-m', 'data_tools.generate', '--split', 'train', '--phase', 'hard-pool', '--run-plan', str(plan.path), '--v0-ready', str(ready_path), '--workers', str(args.workers), '--batch-size', str(args.batch_size)], directory / 'pool.log', plan)
        except RuntimeError:
            time.sleep(min(60, max(5, args.poll_seconds)))
    pool = read_train(pool_path, plan)
    model = directory / 'v0-mlx'
    if not (model / 'conversion.json').is_file():
        if not ensure_idle_v0(plan):
            raise RuntimeError('Training is no longer waiting; cannot claim exclusive GPU use')
        run_command([sys.executable, '-m', 'pastewhat_ranker.export', '--model', ready['checkpoint'], '--output', str(model)], directory / 'export.log', plan)
    mining = directory / 'mining'
    if not (mining / 'selection.json').is_file():
        if not ensure_idle_v0(plan):
            raise RuntimeError('Training is no longer waiting; cannot start mining GPU inference')
        command = [sys.executable, '-m', 'tools.mine_training_pool', '--pool', str(pool_path), '--original-train', str(original_path), '--v0-ready', str(ready_path), '--model', str(model), '--output', str(mining), '--run-plan', str(plan.path), '--gpu-exclusive-confirmation', 'Verified pipeline is waiting for hardening data; state SHA ' + sha256((ROOT / plan.pipeline_directory / 'status.json').read_bytes())]
        if mining.exists():
            command.append('--resume')
        run_command(command, directory / 'mining.log', plan)
    selection = json.loads((mining / 'selection.json').read_text())
    provenance = json.loads((mining / 'provenance.json').read_text())
    if any(selection.get(key) != value or provenance.get(key) != value for key, value in plan.binding().items()):
        raise ValueError('Cached mining output belongs to a different registered run')
    expected_sources = {'v0_handoff_sha256': sha256(ready_path.read_bytes()), 'pool_sha256': sha256(pool_path.read_bytes()), 'original_train_sha256': sha256(original_path.read_bytes()), 'mining_source_sha256': sha256((ROOT / 'tools/mine_training_pool.py').read_bytes())}
    if any(provenance.get(key) != value for key, value in expected_sources.items()):
        raise ValueError('Cached mining source changed; do not silently reuse proposals')
    proposals = read_train(mining / 'proposals.jsonl', plan)
    if sha256((mining / 'proposals.jsonl').read_bytes()) != selection['proposals_sha256']:
        raise ValueError('Mining proposals changed after selection')
    client = TeacherClient(ROOT / 'local/teacher/train/hardcase')
    batches = [proposals[index:index + args.batch_size] for index in range(0, len(proposals), args.batch_size)]
    records = []
    reviews_directory = directory / 'reviews'
    pending = batches[:]
    while pending:
        if coordinator.status()['paused']:
            atomic_json(directory / 'status.json', {**plan.binding(), 'phase': 'post_mining_review_waiting_for_account', 'updated_at': utc_now()})
            time.sleep(min(60, max(5, args.poll_seconds)))
            continue
        retry = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(blind_review, group, client, reviews_directory, plan): group for group in pending}
            for future in as_completed(futures):
                try:
                    records.append(future.result())
                except TeacherError as exc:
                    if 'HTTP 400' in str(exc):
                        raise
                    retry.append(futures[future])
                    atomic_json(directory / 'latest-account-error.json', {**plan.binding(), 'time': utc_now(), 'error': str(exc), 'pending_ids': [row['id'] for row in futures[future]]})
                atomic_json(directory / 'status.json', {**plan.binding(), 'phase': 'post_mining_blind_review', 'reviewed': sum(len(r['accepted']) + len(r['rejected']) for r in records), 'accepted': sum(len(r['accepted']) for r in records), 'rejected': sum(len(r['rejected']) for r in records), 'updated_at': utc_now()})
        pending = retry
        if pending:
            time.sleep(min(60, max(5, args.poll_seconds)))
    tokenizer = ROOT.parent / 'laya-mlx/models/laya-multilingual/tokenizer'
    manifest = publish_mixture(original, pool, records, selection, plan, directory, Preprocessor(tokenizer))
    atomic_json(directory / 'complete.json', manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
