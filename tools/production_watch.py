"""Report production blockers locally without reading examples or calling teachers.

AppleScript notification syntax was checked against Apple's Standard Additions
reference. Notification presentation remains controlled by macOS preferences.
This observer never changes quotas, scheduling, source data or model artifacts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import time

from run_contract import load_run_plan
from tools.publish_pipeline import write_record

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = 'on run argv\n display notification (item 1 of argv) with title (item 2 of argv)\nend run'


def read(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def observe(plan, now):
    local = ROOT / "local"
    rate = read(local / "pi-swe2-account-rate/state.json")
    phases = {name: read(local / directory / plan.run_id / "status.json")
              for name, directory in (("training", "pipeline"),
                                      ("evaluation", "evaluator-release"),
                                      ("publication", "publication"))}
    counts = {}
    for split in ("train", "dev", "calibration", "test"):
        path = (local / "v7" / plan.run_id / (split + ".manifest.json") if split in {"train", "dev"}
                else local / "evaluator-v7" / plan.run_id / split / "production-progress.json")
        record = read(path)
        if record:
            if any(record.get(key) != value for key, value in plan.binding().items()):
                raise ValueError("Production progress belongs to another run")
            counts[split] = record.get("episodes", record.get("retained_unique"))
    summary = {**plan.binding(), "accepted": counts, "max_in_flight": rate.get("max_in_flight"),
               "active_requests": len(rate.get("leases", {})),
               "provider_blocked": bool(rate.get("blocked_reason")),
               "cooldown_seconds": max(0, round(rate.get("cooldown_until", 0) - now)),
               "phases": {name: {key: value for key, value in record.items()
                                  if key in {"phase", "state", "status", "error_type"}}
                          for name, record in phases.items()}}
    events = {}
    if rate.get("blocked_reason"):
        events["provider_blocked"] = "Pi SWE-2 因额度、认证或接口错误暂停新请求。已有数据已保留，请查看训练任务状态。"
    elif summary["cooldown_seconds"] >= 300 and rate.get("pause_reason") != "operator_scheduler_drain":
        events["provider_cooldown"] = "Pi SWE-2 请求进入至少五分钟的退避等待。已有成功请求不会重做，请查看训练任务状态。"
    reduction = rate.get("resource_concurrency_reduction")
    if reduction:
        events["provider_reduced"] = "Pi SWE-2 返回限流，已降低本地并发并退避；生产会遵守接口限制继续执行。"
    for name, record in phases.items():
        if record and any(record.get(key) != value for key, value in plan.binding().items()):
            raise ValueError("Pipeline progress belongs to another run")
        failed = (record.get("error_type") or str(record.get("status", "")).startswith("failed")
                  or record.get("state") == "diagnostic_incomplete_requires_review")
        if failed:
            label = {"training": "学生训练", "evaluation": "独立评估", "publication": "模型发布"}[name]
            events[name + "_failed"] = label + "报告了错误，需要排查后继续。当前状态和原始产物已保留。"
    publication = phases["publication"]
    if publication.get("phase") == "published":
        accepted = publication.get("status") == "accepted_on_frozen_synthetic_benchmark"
        events["published"] = ("模型已通过预定合成验收并发布，PasteWhat 已切换到通过验收的本地模型。" if accepted
                               else "诊断模型和完整指标已发布；部分验收目标未达到，PasteWhat 保持原推理后端。")
    return summary, events


def deliver(message):
    # Only fixed text is passed as argv, never interpolated into AppleScript.
    result = subprocess.run(["/usr/bin/osascript", "-e", SCRIPT, message, "PasteWhat 模型训练"],
                            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=15)
    return {"command_accepted": result.returncode == 0, "exit_code": result.returncode,
            "visible_delivery_verified": False}


def update_alerts(previous, events, now, sender=deliver):
    active = previous.get("active", {})
    outcomes = {}
    for key, message in events.items():
        prior = active.get(key, {})
        if prior.get("command_accepted") or now - prior.get("attempted_at", 0) < 300:
            outcomes[key] = prior
            continue
        try:
            outcome = sender(message)
        except (OSError, subprocess.TimeoutExpired) as error:
            outcome = {"command_accepted": False, "error_type": type(error).__name__,
                       "visible_delivery_verified": False}
        outcomes[key] = {**outcome, "attempted_at": now}
    return {"active": outcomes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-plan", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--announce-start", action="store_true")
    args = parser.parse_args()
    plan = load_run_plan(args.run_plan)
    destination = ROOT / "local/monitor" / plan.run_id
    destination.mkdir(parents=True, exist_ok=True)
    state_path = destination / "notification-state.json"
    with (destination / "monitor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.announce_start:
            result = deliver("SWE-2 生产监看已启动。额度或认证阻断、长退避、已报告的训练与发布错误会在本机提示。")
            write_record(destination / "startup-notification.json", result)
        while True:
            now = time.time()
            try:
                summary, events = observe(plan, now)
            except (OSError, ValueError, TypeError, KeyError) as error:
                summary = {**plan.binding(), "observer_error_type": type(error).__name__}
                events = {"observer_read_error": "训练状态监看无法读取有效状态；请检查本机任务记录。"}
            previous = read(state_path)
            state = update_alerts(previous, events, now)
            state["updated_at"] = datetime.now(timezone.utc).isoformat()
            write_record(state_path, state)
            write_record(destination / "status.json", {**summary, "updated_at": state["updated_at"],
                                                      "active_notifications": list(events)})
            if events.keys() != previous.get("active", {}).keys() or args.once:
                print(json.dumps({"updated_at": state["updated_at"], "events": list(events),
                                  "accepted": summary.get("accepted")}, ensure_ascii=False), flush=True)
            if args.once or state["active"].get("published", {}).get("command_accepted"):
                return
            time.sleep(30)


if __name__ == "__main__":
    main()
