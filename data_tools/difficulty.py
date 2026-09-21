"""Describe synthetic difficulty without inferring correctness from surface overlap."""
from collections import Counter
import json


def visible_text_fields(context):
    fields = [context.get('fieldLabel', ''), context.get('selectedText', '')]
    surrounding = context.get('surroundingText', '')
    try:
        focus = json.loads(surrounding)
    except (ValueError, TypeError):
        focus = None
    if isinstance(focus, dict) and focus.get('format') == 'pastewhat-focus-v1':
        fields.extend(focus.get(key, '') for key in ('beforeSelection', 'afterSelection', 'textWindow'))
        fields.extend(focus.get('nearbyText', []))
    else:
        fields.append(surrounding)
    return [field for field in fields if isinstance(field, str) and field]


def describe_difficulty(episodes):
    counts = Counter()
    families = {}
    for episode in episodes:
        if episode['label']['decision'] != 'select':
            continue
        family = families.setdefault(episode['family_id'], Counter())
        counts['select_episodes'] += 1
        family['select_episodes'] += 1
        positive_ids = set(episode['label']['acceptable_ids'])
        positives = [entry for entry in episode['entries'] if entry['id'] in positive_ids]
        texts = visible_text_fields(episode['context'])
        exact = [entry for entry in positives if entry['capabilities'] == ['text'] and entry['text'] and any(entry['text'] in field for field in texts)]
        counts['text_positive_candidates'] += sum(entry['capabilities'] == ['text'] for entry in positives)
        counts['text_positive_candidates_exactly_in_context'] += len(exact)
        counts['select_episodes_with_exact_positive_literal_in_context'] += bool(exact)
        family['select_episodes_with_exact_positive_literal_in_context'] += bool(exact)
        positive_kinds = {entry['kind'] for entry in positives}
        hard = any(entry['id'] not in positive_ids and entry['kind'] in positive_kinds for entry in episode['entries'])
        counts['select_episodes_with_same_kind_negative'] += hard
        family['select_episodes_with_same_kind_negative'] += hard
        counts['multiple_positive_episodes'] += len(positives) > 1
    selected = counts['select_episodes']
    return {'definition': 'Exact case-sensitive complete positive plaintext occurring within one actual visible text field; no whitespace, case or URL normalization. Native JSON field names/app categories/source metadata excluded. File/image summaries excluded from plaintext overlap.', 'counts': dict(counts), 'select_episode_exact_literal_overlap_fraction': counts['select_episodes_with_exact_positive_literal_in_context'] / selected if selected else None, 'same_kind_negative_fraction_of_select': counts['select_episodes_with_same_kind_negative'] / selected if selected else None, 'candidate_count_distribution': dict(Counter(len(row['entries']) for row in episodes)), 'per_family': {family: dict(value) for family, value in families.items()}, 'limits': 'Direct overlap describes an easy lexical cue, not measured model accuracy; lack of an exact match does not prove semantic difficulty. Same-kind negatives use the actual coarse deployed classifier.'}
