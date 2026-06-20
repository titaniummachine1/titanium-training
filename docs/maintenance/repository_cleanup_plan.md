# Repository cleanup plan

Generated: 2026-06-20T12:18:47.045513+00:00

## Summary

- Files inventoried: 47,759
- Tracked: 103
- Delete candidates: 37,171
- Merge candidates: 1

## Delete candidates (proven dead / generated)

- `engine/target/.rustc_info.json` — Build/cache artifact
- `engine/target/bisect-bnd/.rustc_info.json` — Build/cache artifact
- `engine/target/bisect-bnd/CACHEDIR.TAG` — Build/cache artifact
- `engine/target/bisect-bnd/release/.cargo-lock` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/cfg-if-ac4cbf74619636ed/dep-lib-cfg_if` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/cfg-if-ac4cbf74619636ed/invoked.timestamp` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/cfg-if-ac4cbf74619636ed/lib-cfg_if` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/cfg-if-ac4cbf74619636ed/lib-cfg_if.json` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/crossbeam-deque-8d5aaa9d2b93d9ad/dep-lib-crossbeam_deque` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/crossbeam-deque-8d5aaa9d2b93d9ad/invoked.timestamp` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/crossbeam-deque-8d5aaa9d2b93d9ad/lib-crossbeam_deque` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/crossbeam-deque-8d5aaa9d2b93d9ad/lib-crossbeam_deque.json` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/crossbeam-epoch-505bc660980d31b2/dep-lib-crossbeam_epoch` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/crossbeam-epoch-505bc660980d31b2/invoked.timestamp` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/crossbeam-epoch-505bc660980d31b2/lib-crossbeam_epoch` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/crossbeam-epoch-505bc660980d31b2/lib-crossbeam_epoch.json` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/crossbeam-utils-32cac3d172144180/build-script-build-script-build` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/crossbeam-utils-32cac3d172144180/build-script-build-script-build.json` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/crossbeam-utils-32cac3d172144180/dep-build-script-build-script-build` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/crossbeam-utils-32cac3d172144180/invoked.timestamp` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/crossbeam-utils-d13e1358f328b745/run-build-script-build-script-build` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/crossbeam-utils-d13e1358f328b745/run-build-script-build-script-build.json` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/crossbeam-utils-de8d726518dde80a/dep-lib-crossbeam_utils` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/crossbeam-utils-de8d726518dde80a/invoked.timestamp` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/crossbeam-utils-de8d726518dde80a/lib-crossbeam_utils` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/crossbeam-utils-de8d726518dde80a/lib-crossbeam_utils.json` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/either-3c7663f226e53e12/dep-lib-either` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/either-3c7663f226e53e12/invoked.timestamp` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/either-3c7663f226e53e12/lib-either` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/either-3c7663f226e53e12/lib-either.json` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/getrandom-4b5b8cf5959d807d/dep-lib-getrandom` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/getrandom-4b5b8cf5959d807d/invoked.timestamp` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/getrandom-4b5b8cf5959d807d/lib-getrandom` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/getrandom-4b5b8cf5959d807d/lib-getrandom.json` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/ppv-lite86-3021a327cb998790/dep-lib-ppv_lite86` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/ppv-lite86-3021a327cb998790/invoked.timestamp` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/ppv-lite86-3021a327cb998790/lib-ppv_lite86` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/ppv-lite86-3021a327cb998790/lib-ppv_lite86.json` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/rand-a4756db266523731/dep-lib-rand` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/rand-a4756db266523731/invoked.timestamp` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/rand-a4756db266523731/lib-rand` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/rand-a4756db266523731/lib-rand.json` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/rand_chacha-cca1d18f4ded560a/dep-lib-rand_chacha` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/rand_chacha-cca1d18f4ded560a/invoked.timestamp` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/rand_chacha-cca1d18f4ded560a/lib-rand_chacha` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/rand_chacha-cca1d18f4ded560a/lib-rand_chacha.json` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/rand_core-b31a68977701e898/dep-lib-rand_core` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/rand_core-b31a68977701e898/invoked.timestamp` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/rand_core-b31a68977701e898/lib-rand_core` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/rand_core-b31a68977701e898/lib-rand_core.json` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/rayon-b126c1bc915dc7c0/dep-lib-rayon` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/rayon-b126c1bc915dc7c0/invoked.timestamp` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/rayon-b126c1bc915dc7c0/lib-rayon` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/rayon-b126c1bc915dc7c0/lib-rayon.json` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/rayon-core-0243cfa2bdf67d8c/run-build-script-build-script-build` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/rayon-core-0243cfa2bdf67d8c/run-build-script-build-script-build.json` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/rayon-core-07cb8a9bc0ac1b60/build-script-build-script-build` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/rayon-core-07cb8a9bc0ac1b60/build-script-build-script-build.json` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/rayon-core-07cb8a9bc0ac1b60/dep-build-script-build-script-build` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/rayon-core-07cb8a9bc0ac1b60/invoked.timestamp` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/rayon-core-21fd52796f628df6/dep-lib-rayon_core` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/rayon-core-21fd52796f628df6/invoked.timestamp` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/rayon-core-21fd52796f628df6/lib-rayon_core` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/rayon-core-21fd52796f628df6/lib-rayon_core.json` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/titanium-84d5d9bcde932af8/dep-lib-titanium` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/titanium-84d5d9bcde932af8/invoked.timestamp` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/titanium-84d5d9bcde932af8/lib-titanium` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/titanium-84d5d9bcde932af8/lib-titanium.json` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/titanium-9185445f4778f02f/bin-titanium` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/titanium-9185445f4778f02f/bin-titanium.json` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/titanium-9185445f4778f02f/dep-bin-titanium` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/titanium-9185445f4778f02f/invoked.timestamp` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/zerocopy-0198b0917f31e7cd/dep-lib-zerocopy` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/zerocopy-0198b0917f31e7cd/invoked.timestamp` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/zerocopy-0198b0917f31e7cd/lib-zerocopy` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/zerocopy-0198b0917f31e7cd/lib-zerocopy.json` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/zerocopy-4d691612db51ded8/build-script-build-script-build` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/zerocopy-4d691612db51ded8/build-script-build-script-build.json` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/zerocopy-4d691612db51ded8/dep-build-script-build-script-build` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/zerocopy-4d691612db51ded8/invoked.timestamp` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/zerocopy-9a425f68662e6b09/run-build-script-build-script-build` — Build/cache artifact
- `engine/target/bisect-bnd/release/.fingerprint/zerocopy-9a425f68662e6b09/run-build-script-build-script-build.json` — Build/cache artifact
- `engine/target/bisect-bnd/release/build/crossbeam-utils-32cac3d172144180/build-script-build.exe` — Build/cache artifact
- `engine/target/bisect-bnd/release/build/crossbeam-utils-32cac3d172144180/build_script_build-32cac3d172144180.d` — Build/cache artifact
- `engine/target/bisect-bnd/release/build/crossbeam-utils-32cac3d172144180/build_script_build-32cac3d172144180.exe` — Build/cache artifact
- `engine/target/bisect-bnd/release/build/crossbeam-utils-32cac3d172144180/build_script_build-32cac3d172144180.pdb` — Build/cache artifact
- `engine/target/bisect-bnd/release/build/crossbeam-utils-32cac3d172144180/build_script_build.pdb` — Build/cache artifact
- `engine/target/bisect-bnd/release/build/crossbeam-utils-d13e1358f328b745/invoked.timestamp` — Build/cache artifact
- `engine/target/bisect-bnd/release/build/crossbeam-utils-d13e1358f328b745/output` — Build/cache artifact
- `engine/target/bisect-bnd/release/build/crossbeam-utils-d13e1358f328b745/root-output` — Build/cache artifact
- `engine/target/bisect-bnd/release/build/crossbeam-utils-d13e1358f328b745/stderr` — Build/cache artifact
- `engine/target/bisect-bnd/release/build/rayon-core-0243cfa2bdf67d8c/invoked.timestamp` — Build/cache artifact
- `engine/target/bisect-bnd/release/build/rayon-core-0243cfa2bdf67d8c/output` — Build/cache artifact
- `engine/target/bisect-bnd/release/build/rayon-core-0243cfa2bdf67d8c/root-output` — Build/cache artifact
- `engine/target/bisect-bnd/release/build/rayon-core-0243cfa2bdf67d8c/stderr` — Build/cache artifact
- `engine/target/bisect-bnd/release/build/rayon-core-07cb8a9bc0ac1b60/build-script-build.exe` — Build/cache artifact
- `engine/target/bisect-bnd/release/build/rayon-core-07cb8a9bc0ac1b60/build_script_build-07cb8a9bc0ac1b60.d` — Build/cache artifact
- `engine/target/bisect-bnd/release/build/rayon-core-07cb8a9bc0ac1b60/build_script_build-07cb8a9bc0ac1b60.exe` — Build/cache artifact
- `engine/target/bisect-bnd/release/build/rayon-core-07cb8a9bc0ac1b60/build_script_build-07cb8a9bc0ac1b60.pdb` — Build/cache artifact
- `engine/target/bisect-bnd/release/build/rayon-core-07cb8a9bc0ac1b60/build_script_build.pdb` — Build/cache artifact
- … and 37071 more

## Merge / consolidate

- `training/data/handoff.txt` — Superseded by docs/ — merge or remove
