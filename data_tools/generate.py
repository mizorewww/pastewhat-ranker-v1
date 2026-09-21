"""Generate Train/Dev only, then independently label the exact student view.

Example: uv run python -m data_tools.generate --split train --limit 5000
Completed batches are immutable and resumable. Calibration/Test are owned by the
evaluation agent and are deliberately not accepted by this command.
"""

from __future__ import annotations

import argparse
import copy
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import random
import re
import subprocess
import time

from pastewhat_ranker.preprocess import Preprocessor
from run_contract import action_quotas, family_quotas, load_run_plan
from data_tools.content import ContentRegistry, content_fingerprint
from data_tools.authoring import AUTHORING_PROTOCOL, candidate_space, compile_compact_episode, owned_profile
from data_tools.labeling import LABEL_PROTOCOL, VERDICT_LABEL_SYSTEM, derive_candidate_label
from data_tools.scheduling import prioritize_pilot_batches
from data_tools.deployment import authoring_requirement, placement_issue
from data_tools.teacher import TeacherClient, TeacherError, atomic_json, canonical_bytes, sha256, utc_now


ROOT = Path(__file__).resolve().parents[1]
PARTITION_PATH = Path(__file__).with_name("family_partition.json")
PROMPT_VERSION = "teacher-episodes-v6-compact-verdicts"
CACHE_VERSION = "v6"
CANDIDATE_PROJECTION_PATH = ROOT / "tools/context_projection/provenance.json"
SURFACES = {"recipient", "address_bar", "search", "shell_prompt", "code_editor", "chat_composer", "color", "file_path", "text", "unknown"}

GENERATOR_SYSTEM = """Author synthetic clipboard decisions for the supplied operation and fixed real UI
field. Return valid JSON only, with every requested slot exactly once:
{"episodes":[{"slot":"requested id","guidance":["short visible static instruction"],
"selected":"","candidates":["literal copied text",{"file":["fictional.pdf"]},
{"image":[640,480]}]}]}.

Author only these FOUR fields. guidance is 0–2 plausible static sibling labels,
each at most180 characters. It supplies a specific visible task/constraint, not
hidden intent or an answer key. selected is the entire current field value being
replaced; prefer empty. The fixed profile supplies field metadata and any static
capability help. Before and after the selection are empty unless the supplied
profile explicitly defines literal sides. Each candidate is pasted directly as-is:
no cursor movement, quote insertion, combination, editing or invisible wrapper.
Use complete executable code snippets in an empty code editor; a bare return
needs an actual enclosing function in the pasted snippet or visible fixed sides.
Do not emit context, IDs per candidate, payload metadata, captures or labels.
The code supplies IDs, projects real text/file/PNG payloads with deployed Swift,
and budgets the exact view before independent teachers label it.

candidates must contain EXACTLY the planned number, no exact duplicate payloads.
A string is actual text including commands, code, URLs, colors, prose or filenames.
{"file":[...]} represents real file URLs; names must be safe fictional basenames.
{"image":[width,height]} represents PNG bytes with observable dimensions only,
integers1–8192 and area at most16,777,216. Files/images require a visibly suitable
attachment/canvas target, not a normal text/dimensions field. Never assume unseen
image meaning. Do not supply descriptions, base64, secrets or personal data.
Use reserved example.com/example.org domains and fictional entities.

Keep every task within the supplied operation. Variations must change actual
situations and constraints, not only identifiers. Planned scenario_type:
select: clear visible need; one or more candidates directly satisfy it. With at
least two candidates, include same-type hard negatives with meaningful differing
values/scope. multiple_interchangeable_positives requests distinct forms that
satisfy the SAME fully visible need. Never impose a hidden canonical preference.
no_match: clear need within this operation; every candidate violates it.
ambiguous: visible unresolved incompatible intentions within this operation.
insufficient_context: a necessary deciding fact is visibly unavailable.
The latter two are not lists of several explicitly permitted alternatives.

Every distinction needs visible evidence: destination, exact number, direction,
output format, scope or syntax restrictions. Broad requests can have several
usable variants. Harmless extra output/formatting or equivalent flags must not
be treated as negatives without a visible restriction. If contrasting patch and
statistics, visible guidance must say which output is wanted. If contrasting a
commit diff direction, explicitly state the old-to-new transformation. Do not
assume neighboring commits are parent/child without visible evidence. Keep
constraints short enough for actual neighboring static labels. Preserve planned
count, scenario and language on repairs. No labels, rationales or answer keys.
"""

LABEL_SYSTEM = VERDICT_LABEL_SYSTEM

AUTHOR_OPERATION_GUIDANCE = {
    "git_diff_selection": "When comparing commits, guidance must literally name the requested old/source and new/target revisions. Do not write only 'two commits', 'the specified revisions' or 'explicitly named' without their actual values. Candidate contents cannot supply missing user choices. Name-only diffs also need direction when renames can change the emitted names; do not assume reversing them is always equivalent. State patch/stat/name-only output when that distinction is required.",
    "markdown_link": "For a select or no_match syntax task, show the exact literal destination and exact requested visible link label in guidance or selected text. Vary only Markdown syntax, link label and inline/reference form. Do not require guessing a website's contents from a URL, or create different query/fragment destinations as interchangeable answers. Omit fragment anchors entirely from this operation's authoring. If one link only is required, state exactly one link and no other prose or links.",
    "file_copy_destination": "When correctness depends on destination state, actual visible guidance must say whether the destination directory and its named child already exist and whether the request copies the directory itself or only its contents. Do not silently assume that child is absent: cp -R source dest/source nests another source directory if dest/source already exists. State required overwrite and metadata behavior if those distinguish candidates; otherwise do not reject harmless equivalent copy tools by preference. Never add these facts to an already labeled episode.",
}


def build_plan(split, limit, batch_size, phase, *, target_override=None, run_plan=None):
    partition = json.loads(PARTITION_PATH.read_text())
    families = partition["families"][split][:]
    random.Random(42).shuffle(families)
    target = target_override or (run_plan.target(split) if run_plan else partition["targets"][split])
    if phase != "main":
        target = target_override or (run_plan.document["hardening"]["pool_episodes"] if run_plan else 10000)
    quota, remainder = divmod(target, len(families))
    counts = family_quotas(partition, split, target) if run_plan else {family["id"]: quota + (i < remainder) for i, family in enumerate(families)}
    allocated = action_quotas(counts) if run_plan else None
    # Independent full-family permutations prevent language/count shortcuts.
    # Assign exact global action quotas before slicing work into HTTP batches.
    target_select = round(target * 0.7)
    select_base = {family["id"]: int(counts[family["id"]] * 0.7) for family in families}
    for family in families[:target_select - sum(select_base.values())]:
        select_base[family["id"]] += 1
    if allocated:
        select_base = {key: value["select"] for key, value in allocated.items()}
    schedules = {}
    for family in families:
        identifier = family["id"]
        count = counts[identifier]

        def rng(dimension):
            # Independent registered runs and later mining pools have new
            # schedules, not merely renamed IDs from a previous dataset.
            namespace = "sampling-v6" if phase == "main" else f"sampling-v6/{phase}"
            if run_plan:
                namespace += "/" + run_plan.run_id
            return random.Random(int(sha256(f"{namespace}/42/{split}/{identifier}/{dimension}".encode())[:16], 16))

        no_match_count = allocated[identifier]["no_match"] if allocated else round(count * 0.2)
        missing_count = count - select_base[identifier] - no_match_count
        scenarios = ["select"] * select_base[identifier] + ["no_match"] * no_match_count + ["ambiguous"] * (missing_count // 2) + ["insufficient_context"] * (missing_count - missing_count // 2)
        rng("labels").shuffle(scenarios)
        minimum, maximum = candidate_space(identifier)
        candidate_values = list(range(minimum, maximum + 1))
        candidate_counts = candidate_values * (count // len(candidate_values)) + rng("candidate-remainder").sample(candidate_values, count % len(candidate_values))
        rng("candidate-counts").shuffle(candidate_counts)
        english = round(count * 0.6)
        chinese = round(count * 0.3)
        other = count - english - chinese
        other_languages = ["Japanese", "Spanish", "French", "German"]
        languages = ["English"] * english + ["Simplified Chinese"] * chinese + [other_languages[i % 4] for i in range(other)]
        rng("languages").shuffle(languages)
        stylistic = rng("style")
        schedules[identifier] = [{
            "scenario_type": "insufficient_context" if scenario == "ambiguous" and candidate_count == 1 else scenario,
            "candidate_count": candidate_count,
            "context_language": language,
            "multiple_interchangeable_positives": scenario == "select" and candidate_count >= 2 and identifier not in {"http_method", "configuration_boolean"} and stylistic.random() < 0.2,
            "include_explicit_negation": stylistic.random() < 0.25,
        } for scenario, candidate_count, language in zip(scenarios, candidate_counts, languages, strict=True)]
    batches, emitted = [], 0
    for offset in range(0, max(counts.values()), batch_size):
        for family in families:
            plans = []
            for index in range(offset, min(offset + batch_size, counts[family["id"]])):
                if emitted >= limit:
                    break
                namespace = CACHE_VERSION + ("-" + run_plan.run_id if run_plan else "")
                identifier = f"{split}-{phase}-{namespace}-{family['id']}-{index:05d}"
                plans.append({"id": identifier, **schedules[family["id"]][index], "variant_number": index})
                emitted += 1
            if plans:
                batches.append({"batch_id": f"{family['id']}-{offset:05d}-{len(plans):02d}", "family": family, "plans": plans, **({"run_binding": run_plan.binding()} if run_plan else {})})
            if emitted >= limit:
                return batches
    return batches


def validate_generated(value, plans):
    episodes = value.get("episodes", []) if isinstance(value, dict) else []
    if len(episodes) != len(plans):
        raise ValueError("Generated episode count differs from plan")
    by_id = {episode.get("id"): episode for episode in episodes}
    if len(by_id) != len(episodes) or set(by_id) != {plan["id"] for plan in plans}:
        raise ValueError("Generated episode IDs differ from plan")
    output = []
    for plan in plans:
        episode = by_id[plan["id"]]
        if set(episode) != {"id", "context", "capture", "entries"}:
            raise ValueError("Generator introduced non-input fields")
        if len(episode["entries"]) != plan["candidate_count"]:
            raise ValueError("Candidate count differs from plan")
        if episode["context"].get("isSecure") is not False:
            raise ValueError("Secure contexts do not belong in supervised inference data")
        if episode["context"].get("inputSurface") not in SURFACES:
            raise ValueError("Unknown deployment input surface")
        if episode["context"].get("surroundingText"):
            raise ValueError("Raw surroundingText must be empty; author literal capture fragments instead")
        capture, selected = episode.get("capture", {}), episode["context"].get("selectedText", "")
        if selected and ((capture.get("beforeSelection") == selected and capture.get("afterSelection") == "") or (capture.get("afterSelection") == selected and capture.get("beforeSelection") == "")):
            raise ValueError("Synthetic whole-field capture duplicated selectedText in an unchanged fragment; author a new valid selection")
        nearby = episode.get("capture", {}).get("nearbyText")
        if not isinstance(nearby, list) or any(not isinstance(value, str) for value in nearby):
            raise ValueError("capture.nearbyText must be an array of strings")
        if len(nearby) > 4:
            raise ValueError(f"capture.nearbyText has {len(nearby)} strings; at most FOUR static sibling strings are obtainable")
        if any(len(value) > 240 for value in nearby) or sum(len(value) for value in nearby) > 600:
            raise ValueError("Static sibling guidance exceeds the 240-per-string or 600-total limit; author shorter genuine labels")
        if episode["context"].get("hasAccessibility") is not True and any(episode["context"].get(key) for key in ("fieldLabel", "fieldRole", "selectedText", "surroundingText")):
            raise ValueError("No accessibility context may expose field information")
        seen_payloads = set()
        for entry in episode["entries"]:
            if set(entry) != {"id", "payload", "sourceCategory"}:
                raise ValueError("Author candidates must contain only id, sourceCategory and payload; native code derives text/kind/capabilities")
            payload = entry["payload"]
            if not isinstance(payload, dict) or payload.get("type") not in ("text", "file", "image"):
                raise ValueError("Candidate payload must be text, file or image")
            signature = canonical_bytes(payload)
            if signature in seen_payloads:
                raise ValueError("Identical synthetic payloads cannot occupy two clipboard-history slots")
            seen_payloads.add(signature)
            if payload["type"] == "text" and (set(payload) != {"type", "text"} or not isinstance(payload["text"], str) or not 1 <= len(payload["text"]) <= 20000):
                raise ValueError("Text payload must declare only a nonempty literal body")
            if payload["type"] == "file" and (set(payload) != {"type", "names"} or not isinstance(payload["names"], list) or not 1 <= len(payload["names"]) <= 20):
                raise ValueError("File payload must contain 1–20 synthetic basenames")
            if payload["type"] == "image" and (set(payload) != {"type", "width", "height"} or any(type(payload[key]) is not int or not 1 <= payload[key] <= 8192 for key in ("width", "height")) or payload['width'] * payload['height'] > 16_777_216):
                raise ValueError("Image payload must contain only bounded integer width/height")
        # Counter positional shortcuts independently of what the generator did.
        rng = random.Random(int(hashlib.sha256(episode["id"].encode()).hexdigest()[:16], 16))
        rng.shuffle(episode["entries"])
        for i, entry in enumerate(episode["entries"]):
            entry["id"] = f"c{i + 1}"
        output.append(episode)
    return output


def validate_labels(value, episodes, *, require_quoted=False):
    annotations = value.get("labels", []) if isinstance(value, dict) else []
    by_id = {item.get("id"): item for item in annotations}
    if len(annotations) != len(episodes) or len(by_id) != len(annotations) or set(by_id) != {episode["id"] for episode in episodes}:
        raise ValueError("Label IDs differ from visible input IDs")
    for episode in episodes:
        if require_quoted:
            by_id[episode["id"]]["label"] = derive_candidate_label(by_id[episode["id"]], episode)
        label = by_id[episode["id"]].get("label", {})
        if set(label) != {"decision", "acceptable_ids", "abstain_reason"}:
            raise ValueError("Label violates action contract")
        positives = label["acceptable_ids"]
        if not isinstance(positives, list) or len(positives) != len(set(positives)):
            raise ValueError("Acceptable IDs must be a unique list")
        if not set(positives) <= {entry["id"] for entry in episode["entries"]}:
            raise ValueError("Label selected an absent candidate")
        if label["decision"] == "select":
            if not positives or label["abstain_reason"] is not None:
                raise ValueError("Select requires positives and null reason")
        elif label["decision"] == "abstain":
            if positives or label["abstain_reason"] not in {"no_match", "ambiguous", "insufficient_context"}:
                raise ValueError("Abstain requires no positives and an allowed reason")
        else:
            raise ValueError("Unknown decision")
        quoted = by_id[episode["id"]].get("selected_candidates")
        if require_quoted and quoted is None:
            raise ValueError("Teacher omitted the exact selected-text mapping check")
        if quoted is not None:
            candidate_texts = {entry["id"]: entry["text"] for entry in episode["entries"]}
            if not isinstance(quoted, list) or len(quoted) != len(positives) or {item.get("id") for item in quoted} != set(positives):
                raise ValueError("Quoted candidate IDs differ from positive action set")
            if any(item.get("text") != candidate_texts.get(item.get("id")) for item in quoted):
                raise ValueError("Teacher ID/text correspondence is incorrect")
        # Identical plain-text payloads are interchangeable regardless of their
        # opaque ID. Reject incomplete teacher labels instead of repairing them.
        positive_texts = {entry["text"] for entry in episode["entries"] if entry["id"] in positives and entry["capabilities"] == ["text"]}
        if any(entry["text"] in positive_texts and entry["capabilities"] == ["text"] and entry["id"] not in positives for entry in episode["entries"]):
            raise ValueError("Teacher omitted an identical usable plaintext candidate")
    return by_id


def blind_label_consensus(prepared, annotations, *, client, split, phase, request_id):
    shuffled, mappings = [], {}
    for index, episode in enumerate(prepared):
        entries = copy.deepcopy(episode["entries"])
        random.Random(int(episode["preprocessing"]["visible_sha256"][:16], 16) ^ 9017).shuffle(entries)
        mapping = {}
        for position, entry in enumerate(entries):
            opaque = f"x{position + 1}"
            mapping[opaque] = entry["id"]
            entry["id"] = opaque
        identifier = f"v{index + 1}"
        mappings[identifier] = mapping
        shuffled.append({"id": identifier, "context": episode["context"], "entries": entries})
    shuffled.reverse()
    result = client.complete_json(LABEL_SYSTEM, json.dumps({"episodes": shuffled}, ensure_ascii=False), max_tokens=16384, response_format="json_object", phase=f"{split}-{phase}-blind-label", request_id=request_id)
    verification = validate_labels(result.parsed, shuffled, require_quoted=True)
    agreed, disagreements = set(), []
    for index, episode in enumerate(prepared):
        first = annotations[f"e{index + 1}"]["label"]
        second = copy.deepcopy(verification[f"v{index + 1}"]["label"])
        second["acceptable_ids"] = [mappings[f"v{index + 1}"][identifier] for identifier in second["acceptable_ids"]]
        missing_context_reasons = {"ambiguous", "insufficient_context"}
        same_action = first["decision"] == second["decision"] and set(first["acceptable_ids"]) == set(second["acceptable_ids"])
        same_reason_or_same_abstain_bucket = first["abstain_reason"] == second["abstain_reason"] or (first["decision"] == second["decision"] == "abstain" and first["abstain_reason"] in missing_context_reasons and second["abstain_reason"] in missing_context_reasons)
        equal = same_action and same_reason_or_same_abstain_bucket
        if equal:
            agreed.add(episode["id"])
        else:
            disagreements.append({"id": episode["id"], "type": "blind_label_disagreement", "first": first, "second": second, "audit_id": result.audit_id, "visible_sha256": episode["preprocessing"]["visible_sha256"]})
    return agreed, disagreements, result


def recover_author_request(client, batch, pending, repair, default_user):
    """Reuse a completed in-flight author response with its exact original body.

    A scheduler drain may occur between author/label/audit phases. Before a new
    attempt checkpoint existed, the sanitized request audit is the authoritative
    record of its validation feedback and parameters. No generated input is edited.
    """
    request_id = f"{batch['batch_id']}-g{repair}"
    wanted = {row['id'] for row in pending}
    candidates = []
    for path in client.audit_dir.glob('*.json'):
        audit = json.loads(path.read_text())
        if audit.get('request_id') != request_id or audit.get('status') != 'success':
            continue
        messages = audit.get('request', {}).get('messages', [])
        if len(messages) != 2 or messages[0].get('content') != GENERATOR_SYSTEM:
            continue
        try:
            user = json.loads(messages[1]['content'])
        except (ValueError, KeyError):
            continue
        if {row.get('id') for row in user.get('plans', [])} == wanted and user.get('attempt') == repair:
            candidates.append((audit.get('started_at', ''), messages[1]['content'], audit['audit_id']))
    if candidates:
        _, user, audit_id = max(candidates)
        return user, audit_id
    return default_user, None


def generate_batch(batch, *, client, preprocessor, split, phase, batch_dir, registry=None, run_plan=None):
    from data_tools.audit import AUDIT_SYSTEM, review_group

    if run_plan:
        run_plan.verify_unchanged()
    output_path = batch_dir / f"{batch['batch_id']}.json"
    contract_hash = sha256(canonical_bytes({"prompt": PROMPT_VERSION, "compact_authoring_sha256": sha256(Path(__file__).with_name("authoring.py").read_bytes()), "operation_requirements_sha256": sha256(canonical_bytes(AUTHOR_OPERATION_GUIDANCE)), "author_prompt": sha256(GENERATOR_SYSTEM.encode()), "label_prompt": sha256(LABEL_SYSTEM.encode()), "audit_prompt": sha256(AUDIT_SYSTEM.encode()), "partition": sha256(PARTITION_PATH.read_bytes()), "native_projection_provenance": sha256((ROOT / "tools/context_projection/provenance.json").read_bytes()), "candidate_projection_provenance": sha256(CANDIDATE_PROJECTION_PATH.read_bytes()), "preprocess": preprocessor.manifest(), "batch": batch}))
    accepted, usage, rejected, starting_attempt = {}, {}, [], 0
    if output_path.is_file():
        stored = json.loads(output_path.read_text())
        if stored.get("contract_sha256") != contract_hash:
            stored = migrate_authoring_cache(stored, batch, split, preprocessor, contract_hash, batch_dir)
            atomic_json(output_path, stored)
        complete = {episode["id"] for episode in stored["episodes"]} == {plan["id"] for plan in batch["plans"]}
        if complete and all(episode.get("provenance", {}).get("family_review_audit_id") and episode.get("provenance", {}).get("blind_label_audit_id") for episode in stored["episodes"]):
            return stored
        # Restore once through the partial branch below, so migrated rejections
        # are not counted twice when a previously complete batch loses a slot.
        atomic_json(output_path.with_suffix(".partial.json"), stored)
        output_path.unlink()
    partial_path = output_path.with_suffix(".partial.json")
    if partial_path.is_file():
        partial = json.loads(partial_path.read_text())
        if partial.get("contract_sha256") != contract_hash:
            partial = migrate_authoring_cache(partial, batch, split, preprocessor, contract_hash, batch_dir)
            atomic_json(partial_path, partial)
        if partial.get("contract_sha256") == contract_hash:
            accepted.update({episode["id"]: episode for episode in partial.get("episodes", [])})
            usage.update(partial.get("usage", {}))
            rejected.extend(partial.get("rejected", []))
            starting_attempt = partial.get("attempts_completed", 0)
    last_error = ""
    for repair in range(starting_attempt, starting_attempt + 8):
        if run_plan:
            run_plan.verify_unchanged()
        pending = [plan for plan in batch["plans"] if plan["id"] not in accepted]
        if not pending:
            break
        try:
            # Negative lists caused the author to copy reserved operations into
            # contexts. It receives only the positively stated target; the
            # independent auditor retains the full partition and exclusions.
            positive_operation = batch["family"]["operation"].split(";")[0].split(", excluding")[0].split(" without anchors")[0]
            author_family = {"id": batch["family"]["id"], "operation": positive_operation}
            field_fixture = owned_profile(batch["family"]["id"])
            user = json.dumps({"operation_family": author_family, "field_fixture": field_fixture, "operation_requirements": AUTHOR_OPERATION_GUIDANCE.get(batch["family"]["id"], ""), "plans": pending, "attempt": repair, "previous_validation_findings": last_error, "instructions": "Each episode uses slot equal to its plan id. Exercise only this operation in the fixed field. Preserve scenario_type and exact candidate count. Return only compact episode fields."}, ensure_ascii=False)
            user, recovered_audit_id = recover_author_request(client, batch, pending, repair, user)
            atomic_json(partial_path, {"contract_sha256": contract_hash, "episodes": list(accepted.values()), "usage": usage, "rejected": rejected, "attempts_completed": repair, "pending_author_request": {"request_id": f"{batch['batch_id']}-g{repair}", "user": user, "recovered_success_audit_id": recovered_audit_id}})
            generation = client.complete_json(GENERATOR_SYSTEM, user, max_tokens=24576, response_format="json_object", temperature=1.0 if repair else 0.6, thinking=None if repair else "disabled", phase=f"{split}-{phase}-generate", request_id=f"{batch['batch_id']}-g{repair}")
            usage[f"generation-{generation.audit_id}"] = generation.usage
            raw = generation.parsed.get("episodes", [])
            generated_by_id = {episode.get("slot"): episode for episode in raw if isinstance(episode, dict)}
            prepared = []
            payload_hashes = {}
            findings = []
            for plan in pending:
                try:
                    episode = generated_by_id.get(plan["id"])
                    if episode is None:
                        raise ValueError("Requested episode missing")
                    episode = compile_compact_episode(episode, episode_id=plan["id"], profile=field_fixture, candidate_count=plan["candidate_count"])
                    episode = validate_generated({"episodes": [episode]}, [plan])[0]
                    episode["family_id"] = batch["family"]["id"]
                    from tools.project_context import project_context
                    from tools.project_candidates import project_candidates
                    episode["context"] = project_context(episode["context"], capture=episode["capture"])
                    payload_hashes[episode["id"]] = sha256(canonical_bytes(episode["entries"]))
                    episode["entries"] = project_candidates(episode["entries"])
                    episode = preprocessor.prepare_episode(episode)
                    issue = placement_issue(episode)
                    if issue:
                        raise ValueError(issue)
                    prepared.append(episode)
                except ValueError as exc:
                    findings.append({"id": plan["id"], "finding": str(exc)})
                    rejected.append({"id": plan["id"], "type": "generated_schema_or_visibility", "finding": str(exc), "generation_audit_id": generation.audit_id})
            if not prepared:
                last_error = json.dumps(findings, ensure_ascii=False)
                atomic_json(partial_path, {"contract_sha256": contract_hash, "episodes": list(accepted.values()), "usage": usage, "rejected": rejected, "attempts_completed": repair + 1})
                continue
            # Deliberately exclude operation family, intended scenario type,
            # generation rationale and labels from the independent label request.
            visible = [{"id": f"e{index + 1}", "context": episode["context"], "entries": episode["entries"]} for index, episode in enumerate(prepared)]
            label_result = client.complete_json(LABEL_SYSTEM, json.dumps({"episodes": visible}, ensure_ascii=False), max_tokens=16384, response_format="json_object", phase=f"{split}-{phase}-label", request_id=f"{batch['batch_id']}-l{repair}")
            usage[f"label-{label_result.audit_id}"] = label_result.usage
            annotations = validate_labels(label_result.parsed, visible, require_quoted=True)
            agreed, disagreements, verification = blind_label_consensus(prepared, annotations, client=client, split=split, phase=phase, request_id=f"{batch['batch_id']}-v{repair}")
            usage[f"blind-label-{verification.audit_id}"] = verification.usage
            rejected.extend(disagreements)
            reviews = review_group(prepared, batch["family"], client)
            review_audits = {review["audit_id"] for review in reviews}
            for audit_id in review_audits:
                audit = json.loads((client.audit_dir / f"{audit_id}.json").read_text())
                usage[f"review-{audit_id}"] = audit["response"].get("usage", {})
            for index, (episode, review) in enumerate(zip(prepared, reviews, strict=True)):
                if episode["id"] not in agreed:
                    findings.append({"id": episode["id"], "finding": "Independent blind labels disagreed. Generate a new internally consistent episode respecting the original scenario_type. Keep deliberate ambiguity/no-match when requested; never change a label to force agreement."})
                    continue
                plan = next(plan for plan in batch["plans"] if plan["id"] == episode["id"])
                label = annotations[f"e{index + 1}"]["label"]
                target = plan["scenario_type"]
                actual = label["decision"] if label["decision"] == "select" else label["abstain_reason"]
                matches_target = actual == target or (actual in ("ambiguous", "insufficient_context") and target in ("ambiguous", "insufficient_context"))
                if not matches_target:
                    rejection = {"id": episode["id"], "type": "author_scenario_drift", "requested": target, "independent_label": label, "visible_sha256": episode["preprocessing"]["visible_sha256"]}
                    rejected.append(rejection)
                    findings.append({"id": episode["id"], "finding": f"Verified label was {actual}, but the authoring plan requires {target}. Generate a different episode satisfying the plan; do not edit the old label."})
                    continue
                if not review["accepted"]:
                    rejected.append(review)
                    findings.append({"id": episode["id"], "finding": review["review"]})
                    continue
                issue = placement_issue(episode, label)
                if issue:
                    rejected.append({"id": episode["id"], "type": "literal_paste_placement", "finding": issue, "generation_audit_id": generation.audit_id})
                    findings.append({"id": episode["id"], "finding": issue})
                    continue
                episode["label"] = annotations[f"e{index + 1}"]["label"]
                episode["parent_id"] = episode["id"]
                second_reason = next(item["label"]["abstain_reason"] for item in verification.parsed["labels"] if item["id"] == f"v{index + 1}")
                episode["provenance"] = {
                    **(run_plan.binding() if run_plan else {}),
                    "teacher": "kimi-for-coding",
                    "capture_format": "pastewhat-focus-v1",
                    "candidate_payload_protocol": "native-synthetic-payload-v1",
                    "candidate_fixture_authoring_sha256": payload_hashes[episode["id"]],
                    "candidate_projection_provenance_sha256": sha256(CANDIDATE_PROJECTION_PATH.read_bytes()),
                    "projection_provenance_sha256": sha256((ROOT / "tools/context_projection/provenance.json").read_bytes()),
                    "sampling_protocol": "sampling-v6-independent-family-spaces",
                    "authoring_protocol": AUTHORING_PROTOCOL,
                    "compact_authoring_sha256": sha256(Path(__file__).with_name("authoring.py").read_bytes()),
                    "field_fixture_sha256": sha256(canonical_bytes(field_fixture)),
                    "label_protocol": LABEL_PROTOCOL,
                    "generation_audit_id": generation.audit_id,
                    "generation_phase": phase,
                    "label_audit_id": label_result.audit_id,
                    "generation_model": generation.model,
                    "label_model": label_result.model,
                    "label_visible_sha256": episode["preprocessing"]["visible_sha256"],
                    "family_review_audit_id": review["audit_id"],
                    "observed_family_id": review["review"]["observed_family_id"],
                    "family_review_protocol": "blind-68-operation-classification",
                    "blind_label_audit_id": verification.audit_id,
                    "reason_agreement": episode["label"]["abstain_reason"] == second_reason,
                    "observed_abstain_reasons": [episode["label"]["abstain_reason"], second_reason],
                    "review": "two blind teacher passes agree on action and acceptable IDs after order/ID perturbation; reason agreement recorded separately; independently teacher-reviewed; programmatically validated; not human validated",
                }
                duplicate = registry.claim(episode) if registry is not None else None
                if duplicate:
                    rejected.append(quarantine_duplicate(episode, duplicate, batch_dir))
                    findings.append({"id": episode["id"], "finding": "This complete visible episode duplicates an already accepted slot even after ignoring candidate IDs and order. Generate a new situation with different visible content, not an ID or permutation variant."})
                    continue
                accepted[episode["id"]] = episode
            last_error = json.dumps(findings, ensure_ascii=False)
            atomic_json(partial_path, {"contract_sha256": contract_hash, "episodes": list(accepted.values()), "usage": usage, "rejected": rejected, "attempts_completed": repair + 1})
        except (ValueError, TeacherError) as exc:
            last_error = str(exc)
            rejected.append({"type": "batch_validation", "attempt": repair, "pending_ids": [plan["id"] for plan in pending], "finding": last_error})
            atomic_json(partial_path, {"contract_sha256": contract_hash, "episodes": list(accepted.values()), "usage": usage, "rejected": rejected, "attempts_completed": repair + 1})
            # Authentication/quota failures are not repaired by prompt changes.
            if isinstance(exc, TeacherError) and any(word in str(exc) for word in ("HTTP", "transport", "quota", "account paused")):
                raise
    if len(accepted) != len(batch["plans"]):
        raise ValueError(f"Batch {batch['batch_id']} incomplete after repairs: {last_error}")
    record = {"contract_sha256": contract_hash, "batch_id": batch["batch_id"], "completed_at": utc_now(), "attempts_completed": repair + 1, "usage": usage, "rejected": rejected, "episodes": [accepted[plan["id"]] for plan in batch["plans"]]}
    atomic_json(output_path, record)
    partial_path.unlink(missing_ok=True)
    return record


def migrate_authoring_cache(record, batch, split, preprocessor, contract_hash, batch_dir):
    """Reuse literal numeric captures after the equivalent fragment convenience update.

    Source requests and episode labels/provenance remain unchanged. Each row must
    replay exactly and match the current label and reviewer prompts; this cannot
    silently grandfather weaker annotation rules or different native features.
    """
    from data_tools.replay import ReplayVerifier
    verifier = ReplayVerifier(ROOT, split, preprocessor)
    plans = {plan["id"]: plan for plan in batch["plans"]}
    kept = []
    for episode in record.get("episodes", []):
        try:
            plan = plans[episode["id"]]
            if episode["family_id"] != batch["family"]["id"] or len(episode["entries"]) != plan["candidate_count"]:
                raise ValueError("Cached slot does not match the current sampling plan")
            label = episode["label"]
            observed = label["decision"] if label["decision"] == "select" else label["abstain_reason"]
            if observed != plan["scenario_type"] and {observed, plan["scenario_type"]} - {"ambiguous", "insufficient_context"}:
                raise ValueError("Cached slot does not match its actual action quota")
            verifier.verify(episode)
            issue = placement_issue(episode)
            if issue:
                raise ValueError(issue)
            kept.append(episode)
        except (KeyError, ValueError, OSError) as exc:
            finding = {"id": episode["id"], "type": "incompatible_cached_contract", "finding": str(exc), "content_sha256": content_fingerprint(episode)}
            record.setdefault("rejected", []).append(finding)
            atomic_json(batch_dir / "quarantine" / f"{episode['id']}-{finding['content_sha256']}.json", {"episode": episode, "review": finding, "quarantined_at": utc_now()})
    previous = record.get("contract_sha256")
    record.setdefault("previous_contract_sha256s", []).append(previous)
    record["contract_sha256"] = contract_hash
    record["episodes"] = kept
    record["authoring_migration"] = "Exact native features and original two-label/blind-family responses replayed; no label or feature edits"
    return record


def quarantine_duplicate(episode, duplicate, batch_dir):
    path = batch_dir / "quarantine" / f"{episode['id']}-{duplicate['content_sha256']}.json"
    atomic_json(path, {"reason": "duplicate_content", "duplicate": duplicate, "episode": episode, "quarantined_at": utc_now()})
    return {"id": episode["id"], "type": "duplicate_content", **duplicate}


def reconcile_saved_batches(batch_dir, batches, registry):
    """Recover old duplicate caches by removing only the later unfrozen slot.

    Claims made before publishing each new accepted slot prevent ordinary races.
    This also repairs interrupted/legacy cache state instead of an endless loop
    in which assembly or snapshot validation repeatedly rejects the same file.
    """
    for batch in batches:
        output = batch_dir / f"{batch['batch_id']}.json"
        partial = output.with_suffix(".partial.json")
        for path in (output, partial):
            if not path.is_file():
                continue
            record = json.loads(path.read_text())
            kept, removed = [], []
            for episode in record.get("episodes", []):
                issue = placement_issue(episode)
                if issue:
                    finding = {"id": episode["id"], "type": "literal_paste_placement", "finding": issue, "content_sha256": content_fingerprint(episode)}
                    atomic_json(batch_dir / "quarantine" / f"{episode['id']}-{finding['content_sha256']}.json", {"episode": episode, "review": finding, "quarantined_at": utc_now()})
                    removed.append(finding)
                    continue
                duplicate = registry.claim(episode)
                if duplicate:
                    removed.append(quarantine_duplicate(episode, duplicate, batch_dir))
                else:
                    kept.append(episode)
            if removed:
                record["episodes"] = kept
                record.setdefault("rejected", []).extend(removed)
                record.setdefault("attempts_completed", 0)
                atomic_json(partial, record)
                if path == output:
                    output.unlink()
                break


def assemble(records, split, phase, limit, preprocessor, run_plan=None):
    ordered = sorted(records, key=lambda record: record["batch_id"])
    episodes = [episode for record in ordered for episode in record["episodes"]]
    # Deterministic mixing prevents family-contiguous batches during training.
    random.Random(42).shuffle(episodes)
    dedup = {}
    for episode in episodes:
        digest = content_fingerprint(episode)
        if digest in dedup:
            raise ValueError(f"Duplicate visible episode {episode['id']} and {dedup[digest]}")
        dedup[digest] = episode["id"]
    name = split if phase == "main" else f"{split}_{phase}"
    directory = ROOT / "local/data-production" / run_plan.run_id if run_plan else ROOT / "data"
    output = directory / f"{name}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(canonical_bytes(episode) + b"\n" for episode in episodes)
    temporary = output.with_suffix(".jsonl.tmp")
    temporary.write_bytes(payload)
    temporary.replace(output)
    counts = Counter(episode["label"]["decision"] if episode["label"]["decision"] == "select" else episode["label"]["abstain_reason"] for episode in episodes)
    selected = [episode for episode in episodes if episode["label"]["decision"] == "select"]
    hard_negative = 0
    for episode in selected:
        positives = set(episode["label"]["acceptable_ids"])
        positive_kinds = {entry["kind"] for entry in episode["entries"] if entry["id"] in positives}
        hard_negative += any(entry["id"] not in positives and entry["kind"] in positive_kinds for entry in episode["entries"])
    manifest = {
        **(run_plan.binding() if run_plan else {}),
        "split": split, "phase": phase, "status": "complete" if len(episodes) == limit else "partial", "episodes": len(episodes), "requested_in_this_run": limit,
        "planned_full_split": (run_plan.target(split) if phase == "main" else run_plan.document["hardening"]["pool_episodes"]) if run_plan else json.loads(PARTITION_PATH.read_text())["targets"][split] if phase == "main" else 10000,
        "file": str(output.relative_to(ROOT)), "sha256": sha256(payload), "created_at": utc_now(),
        "family_partition_sha256": sha256(PARTITION_PATH.read_bytes()),
        "prompt_version": PROMPT_VERSION, "preprocessing": preprocessor.manifest(),
        "labels": dict(counts), "candidate_counts": dict(sorted(Counter(len(episode["entries"]) for episode in episodes).items())),
        "families": dict(sorted(Counter(episode["family_id"] for episode in episodes).items())),
        "multiple_positive_episodes": sum(len(episode["label"]["acceptable_ids"]) > 1 for episode in episodes),
        "ambiguous_vs_insufficient_reason_disagreements": sum(episode.get("provenance", {}).get("reason_agreement") is False for episode in episodes),
        "select_with_same_kind_negative": hard_negative, "select_episodes": len(selected),
        "truncated_episodes": sum(episode["preprocessing"]["truncated"] for episode in episodes),
        "teacher": "kimi-for-coding", "teacher_is_rolling": True,
        "usage": {key: sum(phase_usage.get(key, 0) for record in records for phase_usage in record["usage"].values()) for key in ("prompt_tokens", "completion_tokens", "total_tokens")},
        "semantic_rejections": sum(len(record.get("rejected", [])) for record in records),
        "family_reviewed_episodes": sum(bool(episode.get("provenance", {}).get("family_review_audit_id")) for episode in episodes),
        "review": "Teacher-generated and independently teacher-labeled after production preprocessing; independently teacher-reviewed for conceptual-family adherence and deployable fields; schema and invariants programmatically checked; no human validation.",
        "claim_boundary": "Synthetic data only. Deterministic variants are not independent conceptual families. No Calibration/Test examples were accessed by Train/Dev production.",
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    atomic_json(directory / f"{name}.manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "dev"), required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--run-plan", help="Pre-registered production scale and immutable source bindings")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--phase", choices=("main", "hard-pool"), default="main")
    parser.add_argument("--v0-ready", help="Required frozen Dev-selected v0 handoff for a new Train mining pool")
    parser.add_argument("--tokenizer", default=str(ROOT.parent / "laya-mlx/models/laya-multilingual/tokenizer"))
    args = parser.parse_args()
    partition = json.loads(PARTITION_PATH.read_text())
    run_plan = load_run_plan(args.run_plan) if args.run_plan else None
    if run_plan and run_plan.document["teacher_contract_version"] != PROMPT_VERSION:
        raise SystemExit("Registered teacher protocol differs from the generator")
    target = run_plan.target(args.split) if run_plan else partition["targets"][args.split]
    if args.phase == "hard-pool":
        if args.split != "train" or not args.v0_ready:
            raise SystemExit("A hard pool requires Train ownership and a frozen v0 handoff")
        ready_path = (ROOT / args.v0_ready).resolve()
        pipeline_directory = ROOT / run_plan.pipeline_directory if run_plan else ROOT / "local/pipeline"
        if ready_path != pipeline_directory / "ranker-v0-ready.json":
            raise SystemExit("Use the actual training pipeline's v0 handoff")
        ready = json.loads(ready_path.read_text())
        original_train = ROOT / run_plan.data_path("train") if run_plan else ROOT / "data/frozen/train-20000.jsonl"
        if run_plan and any(ready.get(key) != value for key, value in run_plan.binding().items()):
            raise SystemExit("v0 handoff belongs to a different registered run")
        if ready.get("selected_by") != "Dev only" or not original_train.is_file():
            raise SystemExit("The full original Train set and Dev-selected v0 must exist first")
        weight = ROOT / ready["checkpoint"] / "model.safetensors"
        with weight.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != ready["weight_sha256"]:
            raise SystemExit("v0 checkpoint weight differs from its frozen handoff")
        target = run_plan.document["hardening"]["pool_episodes"] if run_plan else 10000
    limit = args.limit or target
    if not 1 <= limit <= target or not 1 <= args.batch_size <= 20 or not 1 <= args.workers <= 8:
        raise SystemExit("Invalid generation size or worker count")
    preprocessor = Preprocessor(args.tokenizer)
    client = TeacherClient(ROOT / "local" / "teacher" / args.split / args.phase)
    batches = build_plan(args.split, limit, args.batch_size, args.phase, target_override=target, run_plan=run_plan)
    batch_dir = ROOT / "local" / "generated" / args.split / (args.phase + "-" + CACHE_VERSION)
    if run_plan:
        batch_dir /= run_plan.run_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    registry_name = f"train-dev-{CACHE_VERSION}" + ("-" + run_plan.run_id if run_plan else "")
    registry = ContentRegistry(ROOT / "local/generated" / (registry_name + "-content.sqlite3"))
    if args.phase == "hard-pool":
        for line in original_train.read_text().splitlines():
            if registry.claim(json.loads(line)):
                raise ValueError("Original frozen training contents conflict with the registry")
    # Frozen pilot bytes win over any later content collision. They are never
    # removed or rewritten by duplicate repair.
    pilot_path = ROOT / run_plan.data_path("pilot") if run_plan else ROOT / "data/frozen/pilot-train-5000.jsonl"
    if pilot_path.is_file():
        for line in pilot_path.read_text().splitlines():
            if registry.claim(json.loads(line)):
                raise ValueError("Frozen pilot reservation conflicts with existing content; preserve snapshot and investigate")
    reconcile_saved_batches(batch_dir, batches, registry)
    if run_plan and args.split == "train" and args.phase == "main":
        accepted = []
        for path in batch_dir.glob("*.json"):
            if path.name != "failures.json":
                accepted.extend(row for row in json.loads(path.read_text()).get("episodes", []) if not placement_issue(row))
        batches = prioritize_pilot_batches(batches, partition, run_plan, accepted)
        # Finish previously dispatched author work first, using its exact body
        # cache, before changing the order of yet-unstarted fixed batches.
        initiated = {path.name.removesuffix('.partial.json').removesuffix('.json') for path in batch_dir.glob('*.json')}
        for path in client.audit_dir.glob('*.json'):
            audit = json.loads(path.read_text())
            try:
                plans = json.loads(audit['request']['messages'][1]['content']).get('plans', [])
            except (KeyError, ValueError, IndexError):
                continue
            if plans and all('-' + run_plan.run_id + '-' in row.get('id', '') for row in plans):
                initiated.add(audit.get('request_id', '').rsplit('-g', 1)[0])
        batches = [batch for batch in batches if batch['batch_id'] in initiated] + [batch for batch in batches if batch['batch_id'] not in initiated]
    records = []
    started = time.monotonic()
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(generate_batch, batch, client=client, preprocessor=preprocessor, split=args.split, phase=args.phase, batch_dir=batch_dir, registry=registry, run_plan=run_plan): batch for batch in batches}
        for future in as_completed(futures):
            batch = futures[future]
            try:
                records.append(future.result())
                completed = sum(len(record["episodes"]) for record in records)
                manifest = assemble(records, args.split, args.phase, limit, preprocessor, run_plan)
                print(json.dumps({"split": args.split, "completed": completed, "target": limit, "labels": manifest["labels"], "elapsed_seconds": round(time.monotonic() - started, 1)}, ensure_ascii=False), flush=True)
            except Exception as exc:
                failures.append({"batch_id": batch["batch_id"], "error": str(exc), "time": utc_now()})
                atomic_json(batch_dir / "failures.json", failures)
                print(json.dumps({"split": args.split, "batch": batch["batch_id"], "error": str(exc)}, ensure_ascii=False), flush=True)
                if isinstance(exc, TeacherError):
                    for pending in futures:
                        pending.cancel()
                    break
    if failures:
        raise SystemExit(f"{len(failures)} generation batches failed; accepted batches and audit records preserved")
    if len(records) != len(batches):
        raise SystemExit("Generation incomplete")


if __name__ == "__main__":
    main()
