# env/ — environment construction

One conda environment per model, because they do not coexist. These scripts are
included because *which versions* turned out to matter more than expected: three
separate failures traced to version combinations rather than to code.

| Script | Environment |
|---|---|
| `build_unicore.sh` | Uni-Core, the backbone for DrugCLIP / BindCLIP / LigUnity / LiTENCLIP / HypSeek |
| `build_conglude_env.sh`, `build_conglude_env2.sh` | ConGLUDe (second attempt is the working one) |
| `build_conplex_env.sh` | ConPLex |
| `build_litenclip_env.sh` | LiTENCLIP |
| `build_sprint_env.sh` | SPRINT |

## Version traps encountered

- **`transformers` newer than the pinned torch breaks imports** — twice, in two
  different environments (ConPLex needs 4.36.2 against torch 2.1; HypSeek needs
  4.44.2 against torch 2.4). The error surfaces far from the cause.
- **Uni-Core's `setup.py` imports torch**, so it must be installed with
  `--no-build-isolation` or the build fails before it starts.
- **The machine's CUDA toolkit and the wheels' CUDA version are independent** —
  a mismatch shows up as a runtime symbol error, not an install error.

Paths inside are from the machine these ran on; they are a record of what was
installed, not a portable installer.
