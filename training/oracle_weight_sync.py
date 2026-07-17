#!/usr/bin/env python3
"""Push promoted weights to Oracle by content hash — skip upload when unchanged."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_TRAINING = _REPO / "training"
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

DEFAULT_CURRENT = _TRAINING / "runs" / "v16" / "net_weights_best.bin"
DEFAULT_PREVIOUS = _TRAINING / "runs" / "v16" / "net_weights_previous.bin"
MIN_WEIGHT_BYTES = 340280
WEIGHT_SCHEMA = "halfpw-sparse-route5-catheat-ws20-cat-v2"
REMOTE_WEIGHTS = "/opt/titanium-game-factory/weights"
REMOTE_INCOMING = "/var/lib/titanium-game-factory/incoming"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_weight(path: Path, tmp_dir: Path, name: str) -> tuple[Path, str]:
    src = path.resolve()
    if not src.is_file():
        raise FileNotFoundError(src)
    dst = tmp_dir / name
    data = src.read_bytes()
    if len(data) < MIN_WEIGHT_BYTES:
        raise RuntimeError(f"{src} too small for oracle ({len(data)} bytes)")
    dst.write_bytes(data)
    return dst, sha256_file(dst)


def ssh_run(host: str, user: str, key: Path, cmd: str) -> str:
    proc = subprocess.run(
        ["ssh", "-i", str(key), f"{user}@{host}", cmd],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or cmd)
    return proc.stdout


def scp_file(host: str, user: str, key: Path, local: Path, remote: str) -> None:
    proc = subprocess.run(
        ["scp", "-i", str(key), str(local), f"{user}@{host}:{remote}"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())


def remote_sha256(host: str, user: str, key: Path, remote_path: str) -> str | None:
    cmd = f"test -f {remote_path} && sha256sum {remote_path} | awk '{{print $1}}' || true"
    out = ssh_run(host, user, key, cmd).strip().lower()
    return out or None


def ensure_remote_weight(
    host: str,
    user: str,
    key: Path,
    local_path: Path,
    expected_sha: str,
) -> bool:
    remote_final = f"{REMOTE_WEIGHTS}/{expected_sha}.bin"
    remote_tmp = f"{remote_final}.tmp"
    existing = remote_sha256(host, user, key, remote_final)
    if existing == expected_sha:
        return False
    ssh_run(
        host,
        user,
        key,
        f"sudo mkdir -p {REMOTE_WEIGHTS} && sudo chown {user}:titanium {REMOTE_WEIGHTS} && sudo chmod 775 {REMOTE_WEIGHTS}",
    )
    scp_file(host, user, key, local_path, remote_tmp)
    actual = remote_sha256(host, user, key, remote_tmp)
    if actual != expected_sha:
        ssh_run(host, user, key, f"rm -f {remote_tmp}")
        raise RuntimeError(f"remote tmp checksum mismatch expected={expected_sha} actual={actual}")
    ssh_run(
        host,
        user,
        key,
        f"mv {remote_tmp} {remote_final} && chmod 644 {remote_final}",
    )
    return True


def push_generation(
    *,
    oracle_host: str,
    user: str,
    key_path: Path,
    token: str,
    url: str,
    current: Path,
    previous: Path,
    epoch: int,
    move_time: float,
    node_budget: int,
) -> dict:
    import urllib.request
    from titanium_training.validation.opening_sanity import assert_opening_sanity

    with tempfile.TemporaryDirectory(prefix="titanium-gen-") as td:
        tmp = Path(td)
        current_path, current_sha = prepare_weight(current, tmp, "current.bin")
        previous_path, previous_sha = prepare_weight(previous, tmp, "prior.bin")
        assert_opening_sanity(current_path)
        assert_opening_sanity(previous_path)
        distinct = current_sha != previous_sha

        uploaded_current = ensure_remote_weight(oracle_host, user, key_path, current_path, current_sha)
        uploaded_previous = False
        if distinct:
            uploaded_previous = ensure_remote_weight(oracle_host, user, key_path, previous_path, previous_sha)

        remote_manifest_raw = ssh_run(
            oracle_host,
            user,
            key_path,
            f"test -f {REMOTE_WEIGHTS}/manifest.json && cat {REMOTE_WEIGHTS}/manifest.json || echo '{{}}'",
        )
        try:
            remote_manifest = json.loads(remote_manifest_raw or "{}")
        except json.JSONDecodeError:
            remote_manifest = {}

        remote_cur = ((remote_manifest.get("current") or {}).get("sha256") or "").lower()
        if remote_cur == current_sha and not uploaded_current and not uploaded_previous:
            return {
                "skipped": True,
                "reason": "remote manifest already has current hash",
                "current_sha256": current_sha,
                "previous_sha256": previous_sha,
            }

        gen_id = f"gen-{__import__('datetime').datetime.now(__import__('datetime').timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        gen_num = int(remote_manifest.get("generation", 0)) + 1
        engine_hash = subprocess.check_output(
            ["git", "-C", str(_REPO), "rev-parse", "HEAD"], text=True
        ).strip()

        manifest = {
            "generation": gen_num,
            "current": {"sha256": current_sha, "filename": f"{current_sha}.bin"},
            "previous": (
                {"sha256": previous_sha, "filename": f"{previous_sha}.bin"}
                if distinct
                else None
            ),
            "engine_sha256": engine_hash,
            "updated_at": __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
        }
        manifest_path = tmp / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        generation = {
            "protocol_version": "titanium-oracle-game-factory/2",
            "generation_id": gen_id,
            "current_deployed_hash": current_sha,
            "prior_deployed_hash": previous_sha,
            "prior_is_distinct": distinct,
            "engine_build_hash": engine_hash,
            "weight_schema": WEIGHT_SCHEMA,
            "search_settings": {
                "move_time_sec": move_time,
                "node_budget": node_budget,
                "engine": "titanium-v17",
            },
            "created_at": manifest["updated_at"],
            "source_promotion_epoch": epoch,
            "generation_seed": int(__import__("time").time()),
            "weights_manifest": manifest,
        }
        gen_json = tmp / "generation.json"
        gen_json.write_text(json.dumps(generation, indent=2) + "\n", encoding="utf-8")

        checksums = tmp / "checksums.sha256"
        lines = []
        for p in (current_path, previous_path, gen_json):
            lines.append(f"{sha256_file(p)}  {p.name}")
        checksums.write_text("\n".join(lines) + "\n", encoding="ascii")

        remote_incoming = f"{REMOTE_INCOMING}/{gen_id}"
        ssh_run(oracle_host, user, key_path, f"sudo mkdir -p {remote_incoming} && sudo chown {user}:{user} {remote_incoming}")
        for name in ("current.bin", "prior.bin", "generation.json", "checksums.sha256"):
            scp_file(oracle_host, user, key_path, tmp / name, f"{remote_incoming}/{name}")

        manifest_remote = f"{REMOTE_WEIGHTS}/manifest.json"
        scp_file(oracle_host, user, key_path, manifest_path, f"{manifest_remote}.tmp")
        ssh_run(
            oracle_host,
            user,
            key_path,
            f"mv {manifest_remote}.tmp {manifest_remote}",
        )

        def post(path: str, body: dict) -> None:
            req = urllib.request.Request(
                f"{url.rstrip('/')}/{path}",
                data=json.dumps(body).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                resp.read()

        post("generation/stage", {"path": remote_incoming})
        post("generation/activate", {"generation_id": gen_id})

        return {
            "skipped": False,
            "generation_id": gen_id,
            "uploaded_current": uploaded_current,
            "uploaded_previous": uploaded_previous,
            "current_sha256": current_sha,
            "previous_sha256": previous_sha,
            "distinct_prior": distinct,
        }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oracle-host", required=True)
    ap.add_argument("--key-path", type=Path, default=Path.home() / ".ssh" / "oracle_titanium.key")
    ap.add_argument("--user", default="ubuntu")
    ap.add_argument("--token", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8765")
    ap.add_argument("--current", type=Path, default=DEFAULT_CURRENT)
    ap.add_argument("--previous", type=Path, default=DEFAULT_PREVIOUS)
    ap.add_argument("--epoch", type=int, default=0)
    ap.add_argument("--move-time", type=float, default=5.0)
    ap.add_argument("--node-budget", type=int, default=200000)
    args = ap.parse_args()
    result = push_generation(
        oracle_host=args.oracle_host,
        user=args.user,
        key_path=args.key_path,
        token=args.token,
        url=args.url,
        current=args.current,
        previous=args.previous,
        epoch=args.epoch,
        move_time=args.move_time,
        node_budget=args.node_budget,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
