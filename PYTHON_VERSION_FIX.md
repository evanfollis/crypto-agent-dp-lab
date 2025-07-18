# Python Version Compatibility Fix Applied

## Previous Issue

The devcontainers were installing Python 3.11.0rc1 (release candidate) from the Ubuntu deadsnakes PPA, which was incompatible with many modern packages that require Python >=3.11. The RC version is considered "older" than the stable 3.11 release by Python's PEP 440 versioning rules.

## Applied Fix

Both `.devcontainer/cuda/Dockerfile` and `.devcontainer/metal/Dockerfile` have been updated to build Python 3.11.10 from source. This ensures a stable, consistent Python version across all development environments.

### Changes Made:

1. **Removed apt package installation of python3.11** (which was installing RC version)
2. **Added Python 3.11.10 source build** with optimization flags
3. **Updated pip installation** to use the built-in pip from Python's ensurepip

### Key Dockerfile Changes:
```dockerfile
# Build Python 3.11.10 from source to ensure we get a stable version
RUN cd /tmp \
 && wget https://www.python.org/ftp/python/3.11.10/Python-3.11.10.tgz \
 && tar -xzf Python-3.11.10.tgz \
 && cd Python-3.11.10 \
 && ./configure --enable-optimizations --with-ensurepip=install \
 && make -j$(nproc) \
 && make altinstall \
 && cd / \
 && rm -rf /tmp/Python-3.11.10*
```

## Next Steps

1. **Rebuild the devcontainer**:
   - In VS Code: Command Palette → "Dev Containers: Rebuild Container"
   - Or manually: `docker-compose down && docker-compose up --build`

2. **Verify Python version**:
   ```bash
   python --version  # Should show: Python 3.11.10
   ```

3. **Reinstall dependencies**:
   ```bash
   poetry lock --no-update
   poetry install --with dev
   ```

## Benefits

- All modern packages (JAX >=0.4.26, NetworkX >=3.2, SciPy >=1.12) can now be installed
- Consistent Python version across all environments
- Optimized Python build for better performance
- No dependency on external PPAs that might change

## Alternative Approaches

If building from source takes too long, consider these alternatives:
1. Use the official Python Docker image as a base layer
2. Use pyenv inside the container
3. Use a different PPA with stable Python versions