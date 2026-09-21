"""Compile small author choices into explicit, observable synthetic UI fixtures.

This adapter uses no label or inferred intent. Owners choose the fixed field
profile before generation. Literal guidance/selection/payloads come from the
author, then the original native projection supplies student-facing metadata.
"""

from __future__ import annotations

import copy


AUTHORING_PROTOCOL = "compact-literal-fixture-v1"


def compile_compact_episode(item, *, episode_id, profile, candidate_count):
    """Return the raw {id,context,capture,entries} native-fixture contract.

    ``item`` contains slot, guidance (0–2 short static strings), selected, and
    candidates. Each candidate is a literal string, {file:[basenames]}, or
    {image:[width,height]}. ``profile`` is an owning split's predetermined actual
    field metadata, optional static capability help, and optional literal sides.
    It must not be selected or changed based on the desired teacher label.
    """
    if set(item) != {"slot", "guidance", "selected", "candidates"}:
        raise ValueError("Compact author must supply only slot/guidance/selected/candidates")
    guidance = item["guidance"]
    if not isinstance(guidance, list) or len(guidance) > 2 or any(not isinstance(line, str) or len(line) > 180 for line in guidance):
        raise ValueError("Author guidance must be 0–2 short static strings, each at most180 characters")
    selected = item["selected"]
    if not isinstance(selected, str) or len(selected) > 1200:
        raise ValueError("Selected whole-field literal must be a bounded string")
    candidates = item["candidates"]
    if not isinstance(candidates, list) or len(candidates) != candidate_count or not 1 <= candidate_count <= 20:
        raise ValueError("Candidate count differs from the predeclared slot; no candidates may be silently dropped")
    permitted = {"applicationCategory", "fieldRole", "fieldLabel", "sourceCategory", "staticGuidance", "beforeSelection", "afterSelection"}
    if set(profile) - permitted:
        raise ValueError("Unknown fixed authoring profile field")
    context = {"applicationCategory": profile["applicationCategory"], "inputSurface": "unknown", "fieldRole": profile["fieldRole"], "fieldLabel": profile["fieldLabel"], "selectedText": selected, "surroundingText": "", "hasAccessibility": True, "isSecure": False}
    capture = {"beforeSelection": profile.get("beforeSelection", ""), "afterSelection": profile.get("afterSelection", ""), "nearbyText": copy.deepcopy(profile.get("staticGuidance", [])) + guidance}
    if len(capture["nearbyText"]) > 4:
        raise ValueError("Fixed help plus author guidance exceeds actual nearby AX capture")
    entries = []
    for index, candidate in enumerate(candidates):
        if isinstance(candidate, str):
            payload = {"type": "text", "text": candidate}
        elif isinstance(candidate, dict) and set(candidate) == {"file"}:
            payload = {"type": "file", "names": candidate["file"]}
        elif isinstance(candidate, dict) and set(candidate) == {"image"}:
            size = candidate["image"]
            if not isinstance(size, list) or len(size) != 2:
                raise ValueError("Image fixture must specify exactly [width,height]")
            payload = {"type": "image", "width": size[0], "height": size[1]}
        else:
            raise ValueError("Candidate must be a text literal, file basenames or image dimensions")
        entries.append({"id": f"c{index + 1}", "sourceCategory": profile["sourceCategory"], "payload": payload})
    return {"id": episode_id, "context": context, "capture": capture, "entries": entries}


def owned_profile(family_id):
    """Profiles for Train/Dev operations only; held-out owners supply their own."""
    command_families = {"file_listing", "literal_text_search", "directory_creation", "file_copy_destination", "disk_usage", "process_inspection", "process_signal", "git_branch_listing", "git_commit_inspection", "git_remote_configuration", "git_diff_selection", "git_tag_version"}
    if family_id in command_families:
        return {"applicationCategory": "development", "fieldRole": "AXTextArea", "fieldLabel": "Shell command", "sourceCategory": "browser", "staticGuidance": ["Paste one complete shell command."]}
    if family_id == "http_method":
        return {"applicationCategory": "development", "fieldRole": "AXTextField", "fieldLabel": "HTTP method", "sourceCategory": "browser"}
    if family_id in {"http_authentication", "http_content_type", "regex_character_class", "configuration_boolean"}:
        return {"applicationCategory": "development", "fieldRole": "AXTextField", "fieldLabel": "Value", "sourceCategory": "browser"}
    if family_id.startswith(("python_", "javascript_", "swift_", "sql_")):
        return {"applicationCategory": "development", "fieldRole": "AXTextArea", "fieldLabel": "Code editor", "sourceCategory": "development", "staticGuidance": ["Paste a complete code snippet."]}
    if family_id == "document_export_format":
        return {"applicationCategory": "mail", "fieldRole": "AXGroup", "fieldLabel": "Attachments", "sourceCategory": "file_management", "staticGuidance": ["Paste files here to attach them."]}
    if family_id == "design_image_dimensions":
        return {"applicationCategory": "creative", "fieldRole": "AXGroup", "fieldLabel": "Image canvas", "sourceCategory": "creative", "staticGuidance": ["Paste a copied image onto this canvas."]}
    if family_id == "design_vector_raster":
        return {"applicationCategory": "creative", "fieldRole": "AXGroup", "fieldLabel": "Design canvas", "sourceCategory": "creative", "staticGuidance": ["Paste SVG markup or a PNG image onto this canvas."]}
    if family_id == "design_color_syntax":
        return {"applicationCategory": "creative", "fieldRole": "AXTextField", "fieldLabel": "Color value", "sourceCategory": "creative"}
    if family_id == "email_recipient_address":
        return {"applicationCategory": "mail", "fieldRole": "AXTextField", "fieldLabel": "To", "sourceCategory": "mail"}
    if family_id == "email_subject_summary":
        return {"applicationCategory": "mail", "fieldRole": "AXTextField", "fieldLabel": "Subject", "sourceCategory": "mail"}
    if family_id.startswith("chat_"):
        return {"applicationCategory": "messaging", "fieldRole": "AXTextArea", "fieldLabel": "Message composer", "sourceCategory": "messaging"}
    if family_id.startswith("spreadsheet_"):
        return {"applicationCategory": "spreadsheet", "fieldRole": "AXTextField", "fieldLabel": "Cell", "sourceCategory": "spreadsheet"}
    if family_id.startswith("url_"):
        return {"applicationCategory": "browser", "fieldRole": "AXTextField", "fieldLabel": "Address", "sourceCategory": "browser"}
    if family_id == "path_absolute_relative":
        return {"applicationCategory": "file_management", "fieldRole": "AXTextField", "fieldLabel": "File path", "sourceCategory": "file_management"}
    if family_id == "contact_postal_address":
        return {"applicationCategory": "browser", "fieldRole": "AXTextArea", "fieldLabel": "Postal address", "sourceCategory": "writing"}
    if family_id in {"writing_concise_summary", "writing_bulleted_outline", "translation_target_language", "markdown_link"}:
        return {"applicationCategory": "writing", "fieldRole": "AXTextArea", "fieldLabel": "Text editor", "sourceCategory": "writing"}
    raise ValueError("No owned Train/Dev authoring profile for this operation")


def candidate_space(family_id):
    # Predeclared before authoring and independent of select/abstain plans. This
    # avoids manufacturing twenty distinct bare values from a tiny vocabulary.
    return (1, 4) if family_id in {"http_method", "configuration_boolean"} else (1, 20)
