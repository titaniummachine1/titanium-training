#!/usr/bin/env python3
"""Evaluation-only Claustrophobia ↔ Titanium cross-play (smoke + paired benches).

Notation is identity (pawn e2, wall e3h). Every move must appear in Claustrophobia's
legal list AND be replayable into Titanium's UCI history. Sync failures abort as
PROTOCOL_ERROR.

Games are written under eval_games/ with DENYLIST.json — never import to labels.db.
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import random
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
_TRAINING = REPO / "training"
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

from engine_session import EngineSession  # noqa: E402

DEFAULT_OUT = Path(__file__).resolve().parent / "eval_games"
DEFAULT_ENGINE = (
    REPO / "engine" / "target-catv5-accepted-03856fe" / "release" / "titanium.exe"
)
OURS = os.environ.get("CLAUSTRO_HTTP", "http://127.0.0.1:9171")
HTTP_MAX_RETRIES = 3
HTTP_RETRY_BACKOFF_SEC = 0.25


class InfrastructureError(RuntimeError):
    """A transport/service failure, never a game result."""

    classification = "infrastructure_error"

# Match benches MUST pass --openings-manifest (dual-validated frozen list).
DEFAULT_OPENINGS: list[tuple[str, ...]] = []


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (ConnectionResetError, ConnectionAbortedError, TimeoutError,
                        urllib.error.URLError, http.client.IncompleteRead)):
        return True
    reason = getattr(exc, "reason", None)
    return isinstance(reason, (ConnectionResetError, ConnectionAbortedError, TimeoutError))


def _request_json(req, timeout: float, method: str) -> dict:
    last = None
    for attempt in range(HTTP_MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError:
            raise
        except Exception as exc:
            if not _is_retryable(exc):
                raise
            last = exc
            if attempt >= HTTP_MAX_RETRIES:
                raise InfrastructureError(
                    f"{method} exhausted {HTTP_MAX_RETRIES} retries: {exc}"
                ) from exc
            print(
                f"HTTP_RETRY method={method} attempt={attempt + 1}/{HTTP_MAX_RETRIES} "
                f"error={exc!r}",
                file=sys.stderr,
            )
            time.sleep(HTTP_RETRY_BACKOFF_SEC * (attempt + 1))
    raise InfrastructureError(f"{method} failed: {last}")


def post(url: str, obj: dict, timeout: float = 600.0) -> dict:
    req = urllib.request.Request(
        url, json.dumps(obj).encode(), {"Content-Type": "application/json"}
    )
    try:
        return _request_json(req, timeout, "POST")
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        try:
            d = json.loads(body)
        except ValueError:
            d = {}
        d.setdefault("error", f"HTTP {e.code}: {body[:200]}")
        return d


def get(url: str, timeout: float = 30.0) -> dict:
    req = urllib.request.Request(url)
    return _request_json(req, timeout, "GET")


class TitaniumSession:
    """Wraps training.engine_session.EngineSession (position/go/ready protocol)."""

    def __init__(self, engine_bin: Path, weights: Path | None, time_sec: float):
        self.time_sec = float(time_sec)
        self.history: list[str] = []
        self.sess = EngineSession(
            "titanium-v17",
            weights,
            threads=1,
            engine_bin=engine_bin,
        )
        if not self.sess.alive():
            raise RuntimeError("Titanium session failed to start")

    def newgame(self) -> None:
        self.history = []
        if not self.sess.sync([]):
            raise RuntimeError("Titanium sync startpos failed")

    def apply(self, mv: str) -> None:
        self.history.append(mv)
        if not self.sess.sync(self.history):
            raise RuntimeError(f"Titanium legality/sync failed at move {mv!r} hist={self.history}")

    def search(self) -> str:
        if not self.sess.sync(self.history):
            raise RuntimeError(f"Titanium sync before search failed hist={self.history}")
        mv = self.sess.go(self.time_sec)
        if not mv:
            raise RuntimeError("Titanium returned no move")
        return mv

    def close(self) -> None:
        self.sess.close()


def _legal_set(st: dict) -> set[str]:
    raw = st.get("legal") or []
    out: set[str] = set()
    for item in raw:
        if isinstance(item, str):
            out.add(item)
        elif isinstance(item, dict) and "notation" in item:
            out.add(str(item["notation"]))
        elif isinstance(item, dict) and "n" in item:
            out.add(str(item["n"]))
    return out


def play_one(
    *,
    titanium_first: bool,
    opening: tuple[str, ...],
    sims: int,
    device: str,
    ti: TitaniumSession,
    move_cap: int = 200,
) -> dict:
    r = post(OURS + "/api/new", {})
    if r.get("error"):
        return {"error": f"api/new failed: {r}", "termination": "PROTOCOL_ERROR"}
    ti.newgame()
    actions: list[str] = []
    disagreements: list[dict] = []
    t0 = time.time()

    for mv in opening:
        st = get(OURS + "/api/state")
        legal = _legal_set(st)
        if legal and mv not in legal:
            return {
                "error": f"opening {mv!r} not legal in Claustrophobia: sample={sorted(legal)[:12]}",
                "termination": "PROTOCOL_ERROR",
                "actions": actions,
            }
        a = post(OURS + "/api/move", {"move": mv})
        if a.get("error"):
            return {
                "error": f"opening {mv!r} rejected: {a}",
                "termination": "PROTOCOL_ERROR",
                "actions": actions,
            }
        try:
            ti.apply(mv)
        except Exception as e:
            return {
                "error": f"Titanium rejected opening history at {mv!r}: {e}",
                "termination": "PROTOCOL_ERROR",
                "actions": actions,
            }
        actions.append(mv)

    for ply in range(len(opening), move_cap):
        st = get(OURS + "/api/state")
        if st.get("terminal"):
            return _finish(st, actions, disagreements, t0, titanium_first, opening)
        legal = _legal_set(st)
        claustro_turn = (ply % 2 == 0) != titanium_first
        if claustro_turn:
            er = post(
                OURS + "/api/engine_move",
                {"sims": str(sims), "device": device},
                timeout=900.0,
            )
            mv = (er.get("lastEngine") or {}).get("notation")
            if not mv:
                return {
                    "error": f"no Claustrophobia move at ply {ply}: {er}",
                    "termination": "PROTOCOL_ERROR",
                    "actions": actions,
                }
            if legal and mv not in legal:
                return {
                    "error": f"Claustrophobia move {mv} not in legal set",
                    "termination": "PROTOCOL_ERROR",
                    "actions": actions,
                }
            try:
                ti.apply(mv)
            except Exception as e:
                return {
                    "error": f"Titanium legality replay failed on Claustrophobia move {mv}: {e}",
                    "termination": "PROTOCOL_ERROR",
                    "actions": actions,
                }
            actions.append({"ply": ply, "side": "claustrophobia", "move": mv})
        else:
            try:
                mv = ti.search()
            except Exception as e:
                return {
                    "error": f"Titanium search failed: {e}",
                    "termination": "PROTOCOL_ERROR",
                    "actions": actions,
                }
            if legal and mv not in legal:
                return {
                    "error": f"Titanium move {mv!r} illegal per Claustrophobia legal set "
                    f"(notation/wall-orientation mismatch?). sample={sorted(legal)[:20]}",
                    "termination": "PROTOCOL_ERROR",
                    "actions": actions,
                    "illegal_move": mv,
                }
            a = post(OURS + "/api/move", {"move": mv})
            if a.get("error") and not a.get("terminal"):
                return {
                    "error": f"Titanium move {mv!r} rejected by Claustrophobia: {a}",
                    "termination": "PROTOCOL_ERROR",
                    "actions": actions,
                    "illegal_move": mv,
                }
            try:
                ti.apply(mv)
            except Exception as e:
                return {
                    "error": f"Titanium self-apply failed after move {mv}: {e}",
                    "termination": "PROTOCOL_ERROR",
                    "actions": actions,
                }
            actions.append({"ply": ply, "side": "titanium", "move": mv})
            if a.get("terminal") or get(OURS + "/api/state").get("terminal"):
                st2 = get(OURS + "/api/state")
                return _finish(st2, actions, disagreements, t0, titanium_first, opening)

    return {
        "error": "move_cap",
        "termination": "ply_cap",
        "actions": actions,
        "disagreements": disagreements,
        "plies": len(actions),
        "seconds": time.time() - t0,
        "titanium_first": titanium_first,
        "opening": list(opening),
    }


def play_one_with_retry(*, max_game_retries: int = 3, **kwargs) -> dict:
    """Replay a failed game from /api/new; never returns partial infra rows."""
    for attempt in range(max_game_retries + 1):
        try:
            return play_one(**kwargs)
        except InfrastructureError as exc:
            if attempt >= max_game_retries:
                raise
            print(
                f"GAME_RETRY attempt={attempt + 1}/{max_game_retries} "
                f"reason=infrastructure_error error={exc}",
                file=sys.stderr,
            )
            # play_one starts with /api/new, so the next attempt is a full replay.
            time.sleep(HTTP_RETRY_BACKOFF_SEC * (attempt + 1))


def _finish(st, actions, disagreements, t0, titanium_first, opening):
    w = st.get("winner")
    try:
        wi = int(w) if w is not None and str(w) not in ("null", "none", "") else None
    except (TypeError, ValueError):
        wi = 0 if str(w).lower() in ("0", "red", "p0") else (1 if w is not None else None)
    if wi is None:
        who = "draw"
        titanium_won = None
    else:
        titanium_won = (wi == 0) if titanium_first else (wi == 1)
        who = "titanium" if titanium_won else "claustrophobia"
    flat = [a["move"] if isinstance(a, dict) else a for a in actions]
    return {
        "winner": wi,
        "winner_side": who,
        "titanium_won": titanium_won,
        "termination": "goal",
        "actions": actions,
        "moves": flat,
        "disagreements": disagreements,
        "plies": len(flat),
        "seconds": time.time() - t0,
        "titanium_first": titanium_first,
        "opening": list(opening),
        "walls_in_moves": sum(1 for m in flat if len(m) == 3 and m[-1] in "hv"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--sims", type=int, default=50)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--time-sec", type=float, default=1.0, help="Titanium go time per move")
    ap.add_argument("--movetime-ms", type=int, default=None, help="deprecated; use --time-sec")
    ap.add_argument("--titanium-bin", type=Path, default=DEFAULT_ENGINE)
    ap.add_argument("--titanium-weights", type=Path, required=True)
    ap.add_argument("--label", required=True, help="candidate|epoch2|smoke")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--openings-manifest",
        type=Path,
        required=True,
        help="Frozen dual-validated openings JSON from build_frozen_openings.py",
    )
    ap.add_argument("--run-id", default="", help="Subdir under out-dir for this clean run")
    args = ap.parse_args()

    manifest = json.loads(args.openings_manifest.read_text(encoding="utf-8"))
    openings_meta = manifest.get("openings") or []
    if len(openings_meta) < args.games:
        print(
            f"HARNESS_FAIL: manifest has {len(openings_meta)} openings, need {args.games}",
            file=sys.stderr,
        )
        return 2
    openings_meta = openings_meta[: args.games]
    manifest_sha = manifest.get("manifest_sha256")
    if not manifest_sha:
        print("HARNESS_FAIL: openings manifest missing manifest_sha256", file=sys.stderr)
        return 2

    run_name = args.run_id or f"{args.label}_vs_claustrophobia"
    out = args.out_dir / run_name
    out.mkdir(parents=True, exist_ok=True)
    denylist = {
        "purpose": "evaluation_only",
        "do_not_import_to_labels_db": True,
        "do_not_train_on": True,
        "source": "claustrophobia_crossplay",
        "titanium_label": args.label,
        "titanium_weights": str(args.titanium_weights),
        "claustrophobia_http": OURS,
        "sims": args.sims,
        "device": args.device,
        "seed": args.seed,
        "openings_manifest": str(args.openings_manifest),
        "openings_manifest_sha256": manifest_sha,
        "openings_version": manifest.get("version"),
        "protocol_aborts_are_harness_failures": True,
    }
    (out / "DENYLIST.json").write_text(json.dumps(denylist, indent=2) + "\n", encoding="utf-8")
    (out / "openings_used.json").write_text(
        json.dumps(
            {
                "manifest_sha256": manifest_sha,
                "openings": openings_meta,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        get(OURS + "/api/state")
    except Exception as e:
        print(f"BLOCKED: cannot reach Claustrophobia at {OURS}: {e}", file=sys.stderr)
        return 2

    time_sec = args.time_sec
    if args.movetime_ms is not None:
        time_sec = max(0.05, args.movetime_ms / 1000.0)

    ti = TitaniumSession(args.titanium_bin, args.titanium_weights, time_sec)
    results_path = out / "results.jsonl"
    summary = {
        "titanium_wins": 0,
        "claustrophobia_wins": 0,
        "draws": 0,
        "protocol_errors": 0,
        "harness_failure": False,
        "games": 0,
        "target_games": args.games,
        "avg_plies": 0.0,
        "total_walls": 0,
        "openings_manifest_sha256": manifest_sha,
        "seed": args.seed,
        "sims": args.sims,
        "time_sec": time_sec,
        "color_split": {"titanium_as_p0": {"w": 0, "l": 0}, "titanium_as_p1": {"w": 0, "l": 0}},
    }
    plies_acc = 0
    try:
        with results_path.open("w", encoding="utf-8") as fh:
            for g in range(args.games):
                titanium_first = (g % 2 == 0)
                meta = openings_meta[g]
                opening = tuple(meta["moves"])
                opening_id = meta["opening_id"]
                row = play_one_with_retry(
                    titanium_first=titanium_first,
                    opening=opening,
                    sims=args.sims,
                    device=args.device,
                    ti=ti,
                )
                row["game_idx"] = g
                row["label"] = args.label
                row["evaluation_only"] = True
                row["opening_id"] = opening_id
                row["opening_seed"] = list(opening)
                row["openings_manifest_sha256"] = manifest_sha
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                summary["games"] += 1
                if row.get("termination") == "PROTOCOL_ERROR":
                    summary["protocol_errors"] += 1
                    summary["harness_failure"] = True
                    print(f"HARNESS_FAIL game {g} opening_id={opening_id}: {row.get('error')}")
                    break
                plies_acc += int(row.get("plies") or 0)
                summary["total_walls"] += int(row.get("walls_in_moves") or 0)
                who = row.get("winner_side")
                key = "titanium_as_p0" if titanium_first else "titanium_as_p1"
                if who == "draw":
                    summary["draws"] += 1
                elif who == "titanium":
                    summary["titanium_wins"] += 1
                    summary["color_split"][key]["w"] += 1
                else:
                    summary["claustrophobia_wins"] += 1
                    summary["color_split"][key]["l"] += 1
                print(
                    f"game {g} opening_id={opening_id} winner={who} plies={row.get('plies')} "
                    f"ti_first={titanium_first} sec={row.get('seconds', 0):.1f}"
                )
    finally:
        ti.close()

    if summary["protocol_errors"] == 0 and summary["games"] == args.games:
        summary["avg_plies"] = plies_acc / max(1, summary["games"])
        summary["clean"] = True
    else:
        summary["clean"] = False
        summary["harness_failure"] = True
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if summary["harness_failure"] or summary["protocol_errors"] > 0:
        return 3
    if summary["games"] != args.games:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
