#!/usr/bin/env python3
"""Overnight pool + NNUE micro-train supervisor.

Watches training logs every ~15s for new errors (flashes pool UI immediately).
Full health check every 5 min (default): pool, train backlog, deploy, parity.

Usage:
    python training/tools/operations/supervise.py --start-pool
    python training/tools/operations/supervise.py --once

Logs: training/data/supervisor.log
Alerts (pool UI): training/data/supervisor_alert.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

from tools.datagen.datagen import DB_PATH, max_game_id, untrained_game_ids  # noqa: E402
from titanium_training.validation.engine_identity import BIN, STAMP, load_expected_stamp  # noqa: E402
from tools.maintenance.manifest import CURRENT_ENGINE, ANCHOR_ENGINE, ANCHOR_RATING, entity_label, load_manifest  # noqa: E402
from titanium_training.training.guards import (  # noqa: E402
    CKPT_DIR,
    DEPLOY_EVERY_GAMES,
    NNUE_LOG,
    artifact_usage,
    enforce_artifact_cap,
    load_guard_state,
)

DATA = ROOT / "training" / "data"
SUP_LOG = DATA / "supervisor.log"
ALERT_JSON = DATA / "supervisor_alert.json"
ALERT_LOG = DATA / "supervisor_alerts.jsonl"
POOL_STARTUP = DATA / "pool_startup.log"
POOL_SCRIPT = ROOT / "training" / "run_swiss_overnight.py"
PARITY_SCRIPT = ROOT / "training" / "parity_check.py"

_TRAIN_FAIL_RE = re.compile(
    r"Training blocked|checkpoint schema|engine validation failed|HARD_CAP|exited [1-9]",
    re.I,
)
_ERROR_LINE_RE = re.compile(
    r"deploy held|hold rebuild|Access is denied|ERROR|FAIL|FATAL|"
    r"file argument must be|Training blocked|engine validation failed|"
    r"remote worker exited|claim failed|parity.*failed|error on game|"
    r"hold drift|preflight failed|pool_preflight|exited \d{2,}",
    re.I,
)
_DEPLOY_STUCK_RE = re.compile(r"deploy held|hold rebuild failed", re.I)


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log(msg: str) -> None:
    line = f"{_ts()} {msg}"
    SUP_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(SUP_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def emit_alert(msg: str, *, level: str = "WARN", source: str = "supervisor") -> None:
    """Push alert to log + JSON for pool UI flash (read by overnight_batch.js)."""
    msg = msg.strip().replace("\n", " ")[:240]
    if not msg:
        return
    payload = {"ts": _ts(), "level": level, "source": source, "msg": msg}
    _log(f"[ALERT:{level}] {msg}")
    DATA.mkdir(parents=True, exist_ok=True)
    ALERT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with open(ALERT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


class LogTail:
    """Track byte offset; return new lines since last poll."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset = path.stat().st_size if path.exists() else 0

    def poll(self) -> list[str]:
        if not self.path.exists():
            return []
        try:
            size = self.path.stat().st_size
            if size < self.offset:
                self.offset = 0
            if size == self.offset:
                return []
            with open(self.path, "rb") as f:
                f.seek(self.offset)
                chunk = f.read(size - self.offset)
            self.offset = size
            text = chunk.decode("utf-8", errors="replace")
            return [ln.strip() for ln in text.splitlines() if ln.strip()]
        except OSError:
            return []


def pool_running() -> bool:
    if sys.platform == "win32":
        try:
            out = subprocess.run(
                [
                    "powershell", "-NoProfile", "-Command",
                    "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | "
                    "Where-Object { $_.CommandLine -match 'run_swiss_overnight' } | "
                    "Measure-Object | Select-Object -ExpandProperty Count",
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            return out.stdout.strip() not in ("", "0")
        except Exception:
            return False
    try:
        out = subprocess.run(["pgrep", "-f", "run_swiss_overnight"], capture_output=True, text=True)
        return out.returncode == 0
    except FileNotFoundError:
        return False


def start_pool() -> bool:
    if not POOL_SCRIPT.exists():
        emit_alert("missing run_swiss_overnight.py", level="FAIL")
        return False
    subprocess.Popen(
        [sys.executable, str(POOL_SCRIPT)],
        cwd=str(ROOT),
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
    )
    _log("started run_swiss_overnight.py")
    emit_alert("supervisor restarted pool process", level="WARN", source="remediate")
    time.sleep(8.0)
    return pool_running()


def tail_train_log(n: int = 40) -> list[str]:
    if not NNUE_LOG.exists():
        return []
    return NNUE_LOG.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]


def recent_train_failures(lines: list[str], window: int = 15) -> tuple[list[str], list[str]]:
    hard, warn = [], []
    for line in lines[-window:]:
        s = line.strip()
        if not s:
            continue
        if "Access is denied" in s and "titanium.exe" in s:
            warn.append(s)
        elif "deploy held" in s or "hold rebuild" in s:
            warn.append(s)
        elif "parity_check failed" in s or "engine validation failed" in s:
            warn.append(s)
        elif _TRAIN_FAIL_RE.search(s):
            hard.append(s)
    return hard, warn


def run_parity() -> tuple[bool, str]:
    if not PARITY_SCRIPT.exists() or not BIN.exists():
        return False, "parity script or titanium.exe missing"
    r = subprocess.run(
        [sys.executable, str(PARITY_SCRIPT)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    text = r.stdout + r.stderr
    m = re.search(r"(\d+)/6 match", text)
    if m and m.group(1) == "6" and r.returncode == 0:
        return True, "6/6"
    return False, text.strip().splitlines()[-1] if text.strip() else f"exit {r.returncode}"


def v15_rating() -> int | None:
    manifest = load_manifest()
    ent = entity_label(CURRENT_ENGINE, "5s")
    info = manifest.get("global_ratings", {}).get(ent)
    return int(info["rating"]) if info else None


def strength_vs_anchor() -> str:
    """v15 Elo vs ti-pure anchor + recent head-to-head from manifest."""
    manifest = load_manifest()
    cur = entity_label(CURRENT_ENGINE, "5s")
    anchor = entity_label(ANCHOR_ENGINE, "5s")
    gr = manifest.get("global_ratings", {})
    v15 = int(gr.get(cur, {}).get("rating", 0)) if cur in gr else None
    anchor_r = int(gr.get(anchor, {}).get("rating", ANCHOR_RATING))
    delta = (v15 - anchor_r) if v15 is not None else None
    h2h = ""
    for m in manifest.get("matchups", {}).values():
        if m.get("a_engine") == CURRENT_ENGINE and m.get("b_engine") == ANCHOR_ENGINE:
            aw, bw = int(m.get("a_wins", 0)), int(m.get("b_wins", 0))
            n = aw + bw
            if n:
                h2h = f" vs-ti-pure {aw}-{bw}/{n}"
            break
    if v15 is None:
        return f"v15=? anchor={anchor_r}{h2h}"
    sign = "+" if delta >= 0 else ""
    return f"v15={v15} ({sign}{delta} vs anchor){h2h}"


def probe_nps(time_sec: float = 2.0) -> str:
    """One-shot genmove NPS on startpos (stderr info json)."""
    if not BIN.exists():
        return "nps=?"
    try:
        r = subprocess.run(
            [
                str(BIN),
                "genmove",
                "--engine",
                CURRENT_ENGINE,
                "--time",
                str(time_sec),
                "--log",
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=max(30, int(time_sec) + 15),
        )
        blob = r.stderr + r.stdout
        last_info = ""
        for line in blob.splitlines():
            if "info json" in line and '"nodes"' in line:
                last_info = line
        if not last_info:
            return "nps=?"
        m_nodes = re.search(r'"nodes"\s*:\s*(\d+)', last_info)
        m_ms = re.search(r'"elapsedMs"\s*:\s*(\d+)', last_info)
        if not m_nodes:
            return "nps=?"
        nodes = int(m_nodes.group(1))
        ms = int(m_ms.group(1)) if m_ms else int(time_sec * 1000)
        nps = int(nodes * 1000 / ms) if ms > 0 else 0
        return f"nps={nps:,} nodes={nodes} @{time_sec}s"
    except Exception as e:
        return f"nps=err({e})"


def watch_logs(tails: dict[str, LogTail]) -> None:
    """Scan new log lines; emit alerts immediately."""
    for name, tail in tails.items():
        for line in tail.poll():
            if not _ERROR_LINE_RE.search(line):
                continue
            level = "FAIL" if _TRAIN_FAIL_RE.search(line) or "FATAL" in line.upper() else "WARN"
            emit_alert(line[:220], level=level, source=name)


def check_deploy_stuck() -> None:
    state = load_guard_state()
    gap = int(state.get("games_since_deploy", 0))
    if gap < DEPLOY_EVERY_GAMES + 8:
        return
    tail = tail_train_log(30)
    if any(_DEPLOY_STUCK_RE.search(ln) for ln in tail[-10:]):
        emit_alert(
            f"DEPLOY STUCK: {gap} trains since last deploy — check nnue_train.log (staging rebuild fix)",
            level="WARN",
            source="deploy",
        )


def remediate(*, pool_ok: bool, pending: int, severity: int) -> None:
    """Best-effort fixes to keep training strength up."""
    if not pool_ok:
        _log("remediate: pool down — attempting restart")
        if not start_pool():
            emit_alert("pool restart failed", level="FAIL", source="remediate")
    if pending > 8:
        emit_alert(f"train backlog {pending} games — train worker may be blocked", level="WARN", source="train")
    check_deploy_stuck()


def check(*, run_parity_check: bool, grace_pool: bool = False) -> tuple[str, int]:
    issues: list[str] = []
    severity = 0

    if not BIN.exists():
        issues.append("titanium.exe missing")
        severity = max(severity, 2)

    stamp = load_expected_stamp()
    if stamp is None:
        issues.append("engine stamp missing")
        severity = max(severity, 1)
    elif not STAMP.exists():
        issues.append("stamp file gone")
        severity = max(severity, 1)

    state = load_guard_state()
    last_trained = int(state.get("last_trained_game_id", 0))
    mx = max_game_id(DB_PATH)
    pending = untrained_game_ids(DB_PATH, last_trained)
    deploy_gap = int(state.get("games_since_deploy", 0))

    cap_ok, cap_msg = enforce_artifact_cap()
    usage = artifact_usage()
    ckpt_mb = usage["checkpoints_bytes"] / 1e6

    if not cap_ok:
        issues.append(cap_msg)
        severity = max(severity, 2)
    elif "WARN" in cap_msg:
        issues.append("soft cap")
        severity = max(severity, 1)

    if len(pending) > 12:
        issues.append(f"train backlog {len(pending)}")
        severity = max(severity, 1)

    pool_ok = pool_running()
    if not pool_ok:
        if grace_pool:
            issues.append("pool starting...")
            severity = max(severity, 1)
        else:
            issues.append("pool not running")
            severity = max(severity, 2)

    log_tail = tail_train_log()
    hard_fails, warn_fails = recent_train_failures(log_tail)
    if hard_fails:
        issues.append(f"recent train errors ({len(hard_fails)})")
        for hf in hard_fails[-3:]:
            emit_alert(hf[:220], level="FAIL", source="nnue_train.log")
        severity = max(severity, 2)
    if warn_fails:
        issues.append(f"deploy blocked ({len(warn_fails)})")
        severity = max(severity, 1)

    parity_ok = True
    parity_msg = "skipped"
    if run_parity_check:
        parity_ok, parity_msg = run_parity()
        if not parity_ok:
            issues.append(f"parity {parity_msg}")
            emit_alert(f"parity check: {parity_msg}", level="WARN", source="parity")
            severity = max(severity, 1)

    rating_s = strength_vs_anchor()
    nps_s = probe_nps(2.0)

    summary = (
        f"pool={'up' if pool_ok else 'DOWN'} "
        f"games={mx} trained={last_trained} pending={len(pending)} "
        f"deploy_gap={deploy_gap} ckpt={ckpt_mb:.0f}MB "
        f"parity={parity_msg} {rating_s}"
    )
    if nps_s:
        summary += f" {nps_s}"
    if issues:
        summary += " | " + "; ".join(issues)

    return summary, severity, pool_ok, len(pending)


def nnue_learning_check() -> tuple[list[str], int]:
    try:
        from titanium_training.training.learning_metrics import collect_learning_report, format_supervisor_lines

        report = collect_learning_report(write_json=True)
        lines = format_supervisor_lines(report)
        sev = 0
        phase = report.get("phase", "")
        if phase == "PLATEAU":
            sev = 1
            since = report.get("games_since_deploy", 0)
            deploy_every = report.get("deploy_every", 32)
            if since > deploy_every + 8:
                emit_alert(
                    report.get("verdict", "PLATEAU")[:220],
                    level="WARN",
                    source="nnue",
                )
        if report.get("pending_train", 0) > 20:
            sev = max(sev, 1)
        return lines, sev
    except Exception as e:
        return [f"NNUE metrics error: {e}"], 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--interval", type=int, default=300, help="Full health check interval sec (default 300)")
    ap.add_argument("--watch-interval", type=int, default=15, help="Log poll interval sec (default 15)")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--start-pool", action="store_true")
    ap.add_argument("--parity-every", type=int, default=3)
    ap.add_argument("--grace-sec", type=int, default=120)
    ap.add_argument("--remediate", action="store_true", default=True)
    ap.add_argument("--no-remediate", action="store_false", dest="remediate")
    args = ap.parse_args()

    tails = {
        "nnue_train.log": LogTail(NNUE_LOG),
        "pool_startup.log": LogTail(POOL_STARTUP),
        "supervisor.log": LogTail(SUP_LOG),
    }

    grace_until = time.time() + max(0, args.grace_sec)
    tick = 0
    next_full = time.time()
    watch_sec = max(5, args.watch_interval)
    full_sec = max(60, args.interval)

    _log(f"supervisor start watch={watch_sec}s full={full_sec}s remediate={args.remediate}")

    while True:
        watch_logs(tails)

        now = time.time()
        if now >= next_full:
            tick += 1
            if args.start_pool and not pool_running():
                start_pool()

            do_parity = args.parity_every > 0 and (tick == 1 or tick % args.parity_every == 0)
            in_grace = now < grace_until
            summary, severity, pool_ok, pending = check(
                run_parity_check=do_parity,
                grace_pool=in_grace,
            )
            level = ("OK", "WARN", "FAIL")[severity]
            _log(f"[{level}] {summary}")

            nnue_lines, nnue_sev = nnue_learning_check()
            nnue_level = ("OK", "WARN", "FAIL")[nnue_sev]
            for ln in nnue_lines:
                _log(f"[{nnue_level}] {ln}")

            if args.remediate and not in_grace:
                remediate(pool_ok=pool_ok, pending=pending, severity=severity)

            if args.once:
                sys.exit(2 if severity >= 2 else (1 if max(severity, nnue_sev) >= 1 else 0))

            # Pool down outside grace: alert but do not kill hidden supervisor (pool is main process).
            if severity >= 2 and not pool_ok and not in_grace:
                emit_alert(f"health FAIL: {summary}", level="FAIL", source="check")

            next_full = now + full_sec

        if args.once:
            break
        time.sleep(watch_sec)


if __name__ == "__main__":
    main()
