# Repository state before bookkeeping cleanup

Captured: 2026-06-20T13:38:22Z

## Git

- Branch: master
- HEAD: fb14f687d49a33e62e230ea4d377b8ab4459862f
- Engine HEAD: 3e551132006d9426f26534310f9f4b0ae3e91c89

## Submodule status

```
git : fatal: no submodule mapping found in .gitmodules for path 'coordinator'
At C:\Users\Terminatort8000\AppData\Local\Temp\ps-script-f8beb243-ab6e-418c-a849-bd1c613054de.ps1:96 char:3
+ $(git submodule status 2>&1 | Out-String)
+   ~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : NotSpecified: (fatal: no submo...h 'coordinator':String) [], RemoteException
    + FullyQualifiedErrorId : NativeCommandError
 

```

## Git status (short)

```
 M .gitignore
 M README.md
 M engine
 M training/README.md
 D training/plan.md
?? .env.example
?? docs/
?? scripts/
?? training/.pytest-temp/
?? training/configs/
?? training/nnue_cli.py
?? training/requirements.txt
?? training/tests/test_oracle_bundle.py
?? training/titanium_training/validation/smoke.py

```

## Top-level folder sizes (MB)

```
.claude: 0,0
.cursor: 0,0
.pytest_cache: 0,0
coordinator: 0,0
docs: 18,4
engine: 12 080,6
KaAiData: 3 003,5
scripts: 0,1
site: 899,7
test-client: 130,8
tools: 943,0
training: 13 445,8

```

## Active dataset manifest hash (expected)

```
31a422f25a8c701ebfa72410f59fab9dff52c2717e30985a3f8e6929be007d02
```

## Backup branch

`backup/pre-bookkeeping-*` created at start of cleanup pass.
