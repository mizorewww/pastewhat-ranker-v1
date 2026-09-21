"""The single teacher/student/deployment input contract.

Prepare before labeling. Only the returned context and entries are teacher-visible;
labels, candidate IDs, family IDs, and audit metadata never enter token features.
All candidates are retained. A second preparation is byte-for-byte idempotent for
the model-visible fields, so saved prepared episodes are safe to encode again.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer

VERSION = "pastewhat-pair-v1"
MAX_PAIR_TOKENS = 1024
CONTEXT_BUDGET = 448
CANDIDATE_BUDGET = 512
METADATA_BUDGET = 64
MAX_CANDIDATES = 20
CONTEXT_FIELDS = (
    "applicationCategory", "inputSurface", "fieldRole", "fieldLabel",
    "selectedText", "surroundingText", "hasAccessibility", "isSecure",
)
APPLICATION_CATEGORIES = frozenset((
    "browser", "development", "terminal", "mail", "messaging", "writing",
    "spreadsheet", "creative", "file_management", "unknown",
))
CAPABILITIES = frozenset(("text", "image", "file", "richText"))
INPUT_SURFACES = frozenset(("unknown", "text", "recipient", "address_bar", "search",
                            "shell_prompt", "code_editor", "chat_composer", "document", "cell",
                            "color", "file_path", "phone"))
CLIP_KINDS = frozenset(("text", "url", "email", "code", "command", "phone", "file", "image", "color"))


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class Preprocessor:
    def __init__(self, tokenizer_path):
        path = Path(tokenizer_path)
        if path.is_dir():
            path = path / "tokenizer.json"
        self.tokenizer = Tokenizer.from_file(str(path))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        self.tokenizer_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        # Use the pretrained tokenizer's own sequence envelope, never guessed IDs.
        empty = self.tokenizer.encode("", add_special_tokens=True).ids
        if len(empty) != 2:
            raise ValueError(f"Expected BOS/EOS tokenizer envelope, found {empty}")
        self.bos_id, self.sep_id = empty
        self.pad_id = self.tokenizer.token_to_id("<pad>")
        if self.pad_id is None:
            raise ValueError("Tokenizer lacks <pad>")

    def tokens(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def _clip(self, text, budget):
        if not isinstance(text, str):
            raise ValueError("Model text fields must be strings")
        ids = self.tokens(text)
        if len(ids) <= budget:
            return text
        # Decode/encode once more because truncated byte-level tokens can split
        # a unicode character; never expose teacher-only tail evidence.
        result = self.tokenizer.decode(ids[:budget], skip_special_tokens=False)
        while len(self.tokens(result)) > budget:
            result = result[:-1]
        return result

    @staticmethod
    def context_body(context):
        return ("Field: " + context["fieldLabel"] + "\nSelected: "
                + context["selectedText"] + "\nSurrounding: "
                + context["surroundingText"])

    @staticmethod
    def metadata(context, entry):
        # Controlled compact metadata; three special tokens also live in its 64.
        return ("Context " + context["applicationCategory"] + " "
                + context["inputSurface"] + " " + context["fieldRole"]
                + (" accessible" if context["hasAccessibility"] else " inaccessible")
                + "\nCandidate " + entry["kind"] + " "
                + ",".join(entry["capabilities"]) + " " + entry["sourceCategory"] + "\n")

    def prepare_episode(self, episode):
        if not isinstance(episode, dict):
            raise ValueError("Episode must be an object")
        source = episode.get("context", {})
        if not isinstance(source, dict):
            raise ValueError("Context must be an object")
        forbidden = {"appName", "bundleID", "pid", "PID", "windowTitle"} & source.keys()
        if forbidden:
            raise ValueError("Native application identity must be projected to categories first")
        category = source.get("applicationCategory", "unknown")
        if category not in APPLICATION_CATEGORIES:
            raise ValueError("Invalid applicationCategory")
        if source.get("inputSurface", "unknown") not in INPUT_SURFACES:
            raise ValueError("Invalid inputSurface")
        for key in ("hasAccessibility", "isSecure"):
            if key in source and not isinstance(source[key], bool):
                raise ValueError(f"{key} must be a JSON boolean")
        if not isinstance(source.get("fieldRole", ""), str):
            raise ValueError("fieldRole must be a string")
        context = {
            "applicationCategory": category,
            "inputSurface": str(source.get("inputSurface", "unknown")),
            "fieldRole": str(source.get("fieldRole", "")),
            "fieldLabel": source.get("fieldLabel", ""),
            "selectedText": source.get("selectedText", ""),
            "surroundingText": source.get("surroundingText", ""),
            "hasAccessibility": bool(source.get("hasAccessibility", True)),
            "isSecure": bool(source.get("isSecure", False)),
        }
        # Metadata must be deployment-visible and short, never an intent channel.
        for key in ("inputSurface", "fieldRole"):
            if len(self.tokens(context[key])) > 12:
                raise ValueError(f"{key} exceeds its metadata budget")
        if context["isSecure"]:
            for key in ("fieldLabel", "selectedText", "surroundingText"):
                context[key] = ""
        original_context = copy.deepcopy(context)
        for key in ("fieldLabel", "selectedText", "surroundingText"):
            context[key] = self._clip(context[key], CONTEXT_BUDGET)
        # Short inputs pass untouched; for long inputs repeatedly trim the longest
        # actual field, so tags/order stay intact and scarce evidence is balanced.
        while len(self.tokens(self.context_body(context))) > CONTEXT_BUDGET:
            key = max(("fieldLabel", "selectedText", "surroundingText"),
                      key=lambda item: len(self.tokens(context[item])))
            excess = len(self.tokens(self.context_body(context))) - CONTEXT_BUDGET
            old = len(self.tokens(context[key]))
            context[key] = self._clip(context[key], max(0, old - max(1, excess)))
        raw_entries = episode.get("entries", [])
        if not isinstance(raw_entries, list):
            raise ValueError("Entries must be an array")
        if len(raw_entries) > MAX_CANDIDATES:
            raise ValueError("A ranker episode supports at most 20 candidates; none may be dropped")
        entries, ids, was_truncated = [], set(), context != original_context
        for raw in raw_entries:
            if not isinstance(raw, dict):
                raise ValueError("Each entry must be an object")
            identifier = raw.get("id")
            if not isinstance(identifier, str) or not identifier or identifier in ids:
                raise ValueError("Candidate IDs must be unique nonempty strings")
            ids.add(identifier)
            if "sourceApp" in raw:
                raise ValueError("Candidate source application must be projected to sourceCategory")
            category = raw.get("sourceCategory", "unknown")
            if category not in APPLICATION_CATEGORIES:
                raise ValueError("Invalid sourceCategory")
            raw_capabilities = raw.get("capabilities", ["text"])
            if not isinstance(raw_capabilities, list) or any(not isinstance(value, str) for value in raw_capabilities):
                raise ValueError("Capabilities must be an array of strings")
            capabilities = sorted(set(raw_capabilities))
            if not capabilities or set(capabilities) - CAPABILITIES:
                raise ValueError("Invalid candidate capabilities")
            kind = raw.get("kind", "text")
            if kind not in CLIP_KINDS:
                raise ValueError("Invalid candidate kind")
            value = self._clip(raw.get("text", ""), CANDIDATE_BUDGET)
            was_truncated |= value != raw.get("text", "")
            entry = {"id": identifier, "text": value, "kind": kind,
                     "capabilities": capabilities, "sourceCategory": category}
            if len(self.tokens(self.metadata(context, entry))) + 3 > METADATA_BUDGET:
                raise ValueError("Metadata exceeds 64-token budget")
            entries.append(entry)
        output = {key: copy.deepcopy(episode[key]) for key in ("id", "family_id", "label") if key in episode}
        output.update(context=context, entries=entries)
        body = {"context": context, "entries": entries}
        output["preprocessing"] = {
            "version": VERSION, "tokenizer_sha256": self.tokenizer_sha256,
            "visible_sha256": hashlib.sha256(canonical_json(body).encode()).hexdigest(),
            "truncated": bool(was_truncated or episode.get("preprocessing", {}).get("truncated", False)),
        }
        return output

    def encode_episode(self, episode):
        prepared = self.prepare_episode(episode)
        context, entries = prepared["context"], prepared["entries"]
        if context["isSecure"]:
            raise ValueError("Secure fields bypass model inference")
        body = self.tokens(self.context_body(context))
        sequences = []
        for entry in entries:
            sequence = ([self.bos_id] + self.tokens(self.metadata(context, entry))
                        + body + [self.sep_id] + self.tokens(entry["text"]) + [self.sep_id])
            if len(sequence) > MAX_PAIR_TOKENS:
                raise AssertionError("Preprocessing violated the pair token budget")
            sequences.append(sequence)
        return {"input_ids": sequences, "candidate_ids": [entry["id"] for entry in entries],
                "prepared": prepared, "pad_id": self.pad_id}

    def manifest(self):
        return {"version": VERSION, "max_pair_tokens": MAX_PAIR_TOKENS,
                "context_budget": CONTEXT_BUDGET, "candidate_budget": CANDIDATE_BUDGET,
                "metadata_and_special_budget": METADATA_BUDGET,
                "max_candidates": MAX_CANDIDATES, "tokenizer_sha256": self.tokenizer_sha256,
                "context_fields": list(CONTEXT_FIELDS), "pooling": "first_token",
                "field_order": ["metadata", "fieldLabel", "selectedText", "surroundingText", "candidateText"],
                "truncation": "prefix_tokens; trim longest context field until total context <=448"}
