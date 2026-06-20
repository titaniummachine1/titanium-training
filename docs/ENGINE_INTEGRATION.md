# Engine integration

The playing engine lives in **`engine/`** — a separate git repository embedded as a submodule.

## Native build (required for training parity)

```powershell
cd engine
$env:RUSTFLAGS = "-C target-cpu=native"
cargo build --release -p titanium
```

Do **not** use suboptimal builds (`TITANIUM_ALLOW_SUBOPTIMAL=1`) for production, overnight pool, benchmarks, or Oracle validation.

Validated binary path used by training:

```text
engine/target/release/titanium.exe   # Windows
engine/target/release/titanium      # Linux
```

## Identity and parity

Training requires a stamped binary and 6/6 Python/engine parity:

```powershell
python training/validate_train_ready.py
python training/parity_check.py
```

Stamp file: `training/data/engine_stamp.json` (gitignored at runtime).

## Bookkeeping freeze

During dataset promotion and Oracle packaging work:

- Do not modify engine source, reset the submodule, or stage submodule pointer changes.
- Local unpushed engine commits remain operator-owned.

## Documentation

- Movegen: `engine/docs/MOVEGEN.md`
- Architecture handoff (NN contract): [ARCHITECTURE.md](ARCHITECTURE.md)
