"""Copy trained net_weights_best.bin to engine live blob only (never frozen)."""
import hashlib
import pathlib
import shutil

REPO = pathlib.Path(__file__).resolve().parents[1]
SRC = REPO / "training" / "runs" / "value_oracle" / "net_weights_best.bin"
DST = REPO / "engine" / "src" / "titanium" / "net_weights.bin"
FROZEN = REPO / "engine" / "src" / "titanium" / "net_weights_frozen.bin"

if not SRC.is_file():
    raise SystemExit(f"missing {SRC}")

frozen_hash = hashlib.sha256(FROZEN.read_bytes()).hexdigest() if FROZEN.is_file() else None
shutil.copy2(SRC, DST)
if frozen_hash and hashlib.sha256(FROZEN.read_bytes()).hexdigest() != frozen_hash:
    raise SystemExit(f"REFUSING: frozen weights were modified at {FROZEN}")
print(f"Deployed live only: {SRC.name} -> {DST} ({SRC.stat().st_size} bytes)")
print(f"Frozen unchanged: {FROZEN.name}")
