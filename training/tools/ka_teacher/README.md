# Ka-AB teacher-source extraction

This copies Ka/Ace script bodies from `reference/ace.html` for offline
comparison or teacher-data work. It does not execute the browser code,
or replace Titanium's live evaluation.

## Batched epoch-15000 NN teacher

The production offline labeler keeps Ka's original JavaScript rules/encoder,
then evaluates the unchanged epoch-15000 network through native ONNX Runtime.
WASM remains the browser backend and the numerical parity oracle.

```powershell
cd training/tools/ka_teacher/native_runtime
npm install
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install onnx numpy pytest
cd ../../../..
.\training\tools\ka_teacher\native_runtime\.venv\Scripts\python training/tools/ka_teacher/export_ka_onnx.py --batch-size 64 --out training/tools/ka_teacher/native_runtime/ka_epoch15000_b64.onnx
.\training\tools\ka_teacher\native_runtime\.venv\Scripts\python -m pytest training/tools/ka_teacher/test_ka_nn_batch_worker.py -q
powershell -NoProfile -ExecutionPolicy Bypass -File training/tools/start_ka_nn_labeling_detached.ps1
```

The detached launcher uses one local DirectML lane, three one-thread local CPU
lanes, and 26 one-thread Oracle CPU lanes by default. Sampling and SQLite writes
remain local and centralized; remote lanes receive only compact move-prefix
batches. Deploy or refresh the Oracle runtime with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File training/tools/ka_teacher/deploy_oracle_pool.ps1
```

The pool stays initialized but automatically pauses while the coordinator state
is `TRAINING`, so local CPU, disk, and memory bandwidth go to the current epoch.
It resumes label production when the training/validation cycle finishes.

Labels are written as source `ka_nn` into existing canonical positions. The
collector updates `training_trigger_state.pending_new_eligible`, so the normal
training coordinator consumes them without a second training pipeline.

## Quick extract (6 core scripts)

```powershell
node training/tools/ka_teacher/extract_ka_ab.js
```

## Full Ace runtime (scripts + weights + WASM)

```powershell
node training/tools/ka_teacher/extract_ace_runtime.js
node training/tools/ka_teacher/ace_harness.mjs --bench --repeats 20
python training/tools/ka_teacher/ace_benchmark.py
```

## Local quarantined teacher labels

```powershell
python training/tools/ka_teacher/ka_local_teacher.py --limit 100
python training/tools/ka_teacher/validate_teacher_sidecar.py --labels training/data/ka_teacher_quarantine/labels.jsonl
```

The default destination for `extract_ka_ab.js` is `work/ka_ab_teacher`. To choose another destination,
pass `--out <directory>`. The extractor fails if a required script id is missing
or occurs more than once, and writes `manifest.json` with source/body SHA-256s.

Required script ids: `engine-core`, `ka-encoder`, `ka-forward`, `ka-solver`,
`ka-engine`, and `ka-ab`.

## Bounded Ka-AB teacher adapter

`ka_ab_teacher.mjs` reads `C:\Users\Terminatort8000\Downloads\ace.html` in
place, evaluates the seven required bodies (including `ka-weights`) in a
Node VM, and runs the beta Ka alpha-beta engine without a browser:

```powershell
node training/tools/ka_teacher/ka_ab_teacher.mjs --nodes 8
node training/tools/ka_teacher/ka_ab_teacher.mjs --moves e2 e8 --nodes 8 --time-ms 0
node training/tools/ka_teacher/ka_ab_teacher.mjs --backend auto --nodes 8 --bench 10
python -m pytest training/tools/ka_teacher/test_ka_ab_teacher.py -q
```

Use `--ace <path>` to select another supplied bundle. `--nodes` is passed as
the AB `maxEvals` bound; `--time-ms` is accepted only as exactly `0` so output
is deterministic and node-bounded. `--backend auto|wasm|js` selects the
existing extracted backend (default `auto`, preferring WASM-SIMD and falling
back to JS); `--backend wasm` fails if WASM cannot initialize. `--bench
<repeats>` repeats the search and adds timing fields to the single JSON
output. The adapter emits one JSON object with schema `ace-ka-ab-teacher-v1`
and never uses `reference/ace.html`.

`extract_ace_runtime.js` additionally exports `ka-forward-wasm`, `ka-forward-webgpu`,
`ka-backend`, `ka-worker`, `ka-weights.json`, and `ka-wasm-bin.json` into
`reference/ka_weights_export/` (gitignored).

## Bounded Ace MCTS teacher prototype

`ace_mcts_teacher.mjs` reads the supplied bundle at
`C:\Users\Terminatort8000\Downloads\ace.html` in place, evaluates the certified
default PUCT MCTS in a Node VM, and emits one JSON label on stdout. The beta
alpha-beta engine is not loaded or selected:

```powershell
node training/tools/ka_teacher/ace_mcts_teacher.mjs --nodes 32
node training/tools/ka_teacher/ace_mcts_teacher.mjs --moves e2 e8 --nodes 64
python -m pytest training/tools/ka_teacher/test_ace_mcts_teacher.py -q
```

The output includes canonical official moves, the side-to-move teacher value,
all legal root priors and visit counts, requested/actual search budget, the
SHA-256 hash of the unchanged Ace bundle, and schema metadata. `--nodes` is the
reproducibility control; `--time-ms 0` is the default so the smoke test is
node-bounded rather than wall-clock bounded.
