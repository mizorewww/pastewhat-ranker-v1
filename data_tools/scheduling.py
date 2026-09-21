"""Order unchanged authoring batches to finish registered pilot strata sooner."""
from collections import Counter
from itertools import combinations

from run_contract import action_quotas, family_quotas


def scenario_bucket(value):
    return value if value in ('select', 'no_match') else 'missing_intent'


def prioritize_pilot_batches(batches, partition, run_plan, accepted=()):
    """Return a permutation only; never alter slots, batch contents or labels.

    Small registered runs have only a few batches per family. For larger runs a
    bounded greedy search avoids combinatorial scheduling cost. Already accepted
    rows reduce deficits; choosing a batch uses only its predeclared plan buckets.
    """
    quotas = action_quotas(family_quotas(partition, 'train', run_plan.document['pilot_episodes']))
    known = {row['id'] for row in accepted}
    completed = Counter((row['family_id'], scenario_bucket(row['label']['decision'] if row['label']['decision'] == 'select' else row['label']['abstain_reason'])) for row in accepted)
    grouped = {}
    for index, batch in enumerate(batches):
        plans = [row for row in batch['plans'] if row['id'] not in known]
        if plans:
            grouped.setdefault(batch['family']['id'], []).append((index, Counter(scenario_bucket(row['scenario_type']) for row in plans), len(plans)))
    priority = set()
    for family, target in quotas.items():
        missing = {bucket: max(0, number - completed[family, bucket]) for bucket, number in target.items()}
        if not any(missing.values()):
            continue
        options = grouped.get(family, [])
        chosen = None
        if len(options) <= 12:
            best = None
            for amount in range(1, len(options) + 1):
                for selection in combinations(options, amount):
                    totals = sum((row[1] for row in selection), Counter())
                    if all(totals[bucket] >= number for bucket, number in missing.items()):
                        key = (sum(row[2] for row in selection), amount, tuple(row[0] for row in selection))
                        if best is None or key < best:
                            best, chosen = key, selection
                if best is not None:
                    break
        if chosen is None:
            chosen = []
            remaining = options[:]
            while any(missing.values()) and remaining:
                best = max(remaining, key=lambda row: (sum(min(row[1][bucket], number) for bucket, number in missing.items()) / row[2], -row[0]))
                chosen.append(best)
                remaining.remove(best)
                for bucket in missing:
                    missing[bucket] = max(0, missing[bucket] - best[1][bucket])
        priority.update(row[0] for row in chosen)
    return [row for index, row in enumerate(batches) if index in priority] + [row for index, row in enumerate(batches) if index not in priority]
