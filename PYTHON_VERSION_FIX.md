# Python Version Compatibility Fix Required

## Current Issue

This container is running Python 3.11.0rc1 (release candidate), which is incompatible with many modern packages that require Python >=3.11. The RC version is considered "older" than the stable 3.11 release by Python's PEP 440 versioning rules.

## Impact

The following packages cannot be installed with Python 3.11.0rc1:
- NetworkX >=3.2 
- SciPy >=1.12
- JAX >=0.4.26
- Many other modern packages

## Required Fix

To use the full crypto DP stack, you need to update to a stable Python 3.11.x version:

### Option 1: Update Dev Container
In `.devcontainer/metal/Dockerfile`:
```dockerfile
# Replace:
RUN conda install -y python=3.11

# With:
RUN conda install -y python=3.11.8
```

### Option 2: Use GitHub Actions CI
The CI workflow has been configured to use Python 3.11.8:
```yaml
- name: Set up Python
  uses: actions/setup-python@v5
  with:
    python-version: "3.11.8"
```

### Option 3: Local Development
```bash
# macOS with Homebrew
brew install python@3.11
brew unlink python@3.12
brew link --overwrite python@3.11

# Or use pyenv
pyenv install 3.11.8
pyenv local 3.11.8
```

## Temporary Workaround

Until the Python version is updated, the project uses older package versions that are compatible with Python 3.11.0rc1. Once you update to Python 3.11.8+, update `pyproject.toml` with the recommended versions from the build instructions.

## Verification

After updating Python, verify with:
```bash
python --version  # Should show 3.11.8 or higher (not rc1)
poetry lock --no-update
poetry install --with dev
```