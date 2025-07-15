
## ⬛ DP‑Crypto Lab – Agent Onboarding Brief (v0.1)

### 0 Mission (why this repo exists)

> Build a **differentiable‑programming / reinforcement‑learning framework** that trades crypto assets live.
> Emphasise reproducibility, audit‑grade logging, and containerised portability across
> • **macOS (M4 Pro, Metal/MPS)** for rapid iteration
> • **WSL 2 + RTX 5090 CUDA 12.8** for full‑scale training & CI.

### 1 Golden Rules for Code‑Gen Agents

1. **One PR = One atomic concern** (feature, refactor, or infra task).
2. **No global site‑package edits.** Every change lives inside the dev‑container image.
3. **Respect the scaffolds below.** If a file or directory is not mentioned, leave it untouched.
4. **Always add/adjust tests** when you change behaviour or introduce new modules.
5. **Run `./scripts/preflight.sh` locally**; commit only if it exits 0.

### 2 Repo Skeleton (generate if missing)

```
crypto‑dp‑lab/
├── .devcontainer/
│   ├── cuda/      # Dockerfile + devcontainer.json
│   └── metal/
├── env/           # conda yaml templates (cpu.yml, cuda.yml)
├── src/
│   ├── dp_lab/    # python package root (leave empty for now)
│   └── tests/
├── data/          # READ‑ONLY mountpoint (no raw files committed)
├── secrets/       # *.enc age‑encrypted files (never plaintext)
├── scripts/
│   ├── preflight.sh
│   └── format_check.sh
├── .github/
│   └── workflows/ (ci.yml template stub)
├── .pre‑commit‑config.yaml
├── pyproject.toml   (Poetry, empty deps list)
└── README.md        (light placeholder)
```

### 3 Immediate TODO Backlog (ordered)

| ID       | Task                                                                                                                           | Output File(s)             | Constraints                                        |
| -------- | ------------------------------------------------------------------------------------------------------------------------------ | -------------------------- | -------------------------------------------------- |
| **T‑00** | Add **`.pre‑commit‑config.yaml`** (black, isort, flake8, case‑collision‑lint).                                                 | `.pre‑commit‑config.yaml`  | Pin tool versions; enable on `python/**` & `src/`. |
| **T‑01** | Scaffold **`pyproject.toml`** for Poetry 3.11, **no runtime deps yet**.                                                        | `pyproject.toml`           | Leave `[tool.poetry.dependencies]` empty.          |
| **T‑02** | Generate **`cuda.Dockerfile`** (`FROM nvidia/cuda:12.8.0‑devel‑ubuntu22.04`) and matching `cuda.devcontainer.json`.            | `.devcontainer/cuda/*`     | Add runArgs `"--gpus","all"`, mount `/opt/data`.   |
| **T‑03** | Generate **`metal.Dockerfile`** (Ubuntu jammy‑slim + Python 3.11) + `metal.devcontainer.json`.                                 | `.devcontainer/metal/*`    | No GPU flags.                                      |
| **T‑04** | Create **`scripts/preflight.sh`** implementing: driver‑match check, `torch.cuda`/MPS availability, poetry install test.        | `scripts/preflight.sh`     | Fail‑fast pattern; POSIX‑sh compatible.            |
| **T‑05** | Stub **`ci.yml`** with matrix `{ os:[macos‑14,ubuntu‑22.04], gpu:[cpu,cuda] }`, job installs Poetry, runs pre‑commit + pytest. | `.github/workflows/ci.yml` | GPU job marked `if: matrix.gpu == 'cuda'`.         |
| **T‑06** | Add **`format_check.sh`** (runs black, isort, flake8) + hook in pre‑commit.                                                    | `scripts/format_check.sh`  | Exits non‑zero on diff.                            |

> **IMPORTANT** – Agents must *stop after T‑06*. Leave DP logic, data ingestion, and trading code for later PRs.

### 4 Style & Naming

* **Python 3.11** only.
* Strict type hints (`mypy --strict` in later sprints).
* Modules snake\_case, classes PascalCase, constants UPPER\_SNAKE.
* Commit messages: `<scope>: <imperative summary>`; body ≤ 72 chars/line.

### 5 Secrets & Data

* **Never output plaintext keys.** Use `sops‑age` encrypted placeholders.
* **No raw datasets in Git.** Access via `/opt/data` bind‑mount (volume defined outside repo).
* Unit tests must run offline with **synthetic fixtures**.

### 6 How to Validate before PR

```bash
# inside dev‑container
poetry install --with dev
pre‑commit install   # first time only
./scripts/preflight.sh
pytest -q
```

### 7 Escalation Path

If an agent is uncertain, it must:

1. Leave a `TODO:` comment in code **and**
2. Emit an inline question in the PR description (GitHub UI) asking for clarification.

### 8 Forbidden Actions

* Auto‑upgrading system packages in Dockerfiles (`apt upgrade -y`) – pin versions instead.
* Adding new dependencies without `poetry add` + lock update.
* Pushing to `main`—all changes must come via PR.

---
