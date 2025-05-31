# PyCharm Dependency Installation Setup Guide

## Understanding the Dependency Files

* **`requirements.txt`** → Production dependencies only
* **`requirements-dev.txt`** → Development dependencies (includes `requirements.txt`)
* **`pyproject.toml`** → Project metadata and package configuration
* **`scripts/setup.sh`** → Automated setup script

---

## Method 1: Automated Setup (Recommended)

### 1.1 First, Create the Missing Requirements Files

Create `requirements.txt`:

```txt
# Core FastAPI and ASGI
fastapi==0.104.1
uvicorn[standard]==0.24.0.post1
pydantic==2.4.2
pydantic-settings==2.0.3
```

Then run:

```bash
./scripts/setup.sh
```

This script will:

* ✅ Check Python version
* ✅ Create virtual environment
* ✅ Install all dependencies
* ✅ Set up pre-commit hooks
* ✅ Create `.env` file

---

## Method 2: Manual PyCharm Setup

### 2.1 Create Virtual Environment

Go to:

* **File → Settings → Project → Python Interpreter**
* **Add Interpreter → Virtualenv Environment → New**

  * Location: `./venv`
  * Base interpreter: Python 3.11+
  * Click OK

### 2.2 Install Dependencies

```bash
# Activate virtual environment
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements-dev.txt
pip install -e .
```

### 2.3 Verify Installation

```bash
pip list
python -c "import fastapi, motor, redis, pydantic; print('✅ All major dependencies installed')"
```

---

## Method 3: Using `pyproject.toml`

```bash
# Production only
pip install .

# With dev dependencies
pip install .[dev]

# Editable install
pip install -e .[dev]
```

---

## PyCharm Configuration After Installation

### 3.1 Configure Python Interpreter

Verify `./venv` is selected and all packages are visible.

### 3.2 Mark Source Directories

Right-click `src/` → Mark Directory as → Sources Root

### 3.3 Set Environment Variables

* **Run → Edit Configurations**
* Load from `.env` or manually set:

  ```
  ENVIRONMENT=development
  DEBUG=true
  PYTHONPATH=src
  ```

---

## Testing the Setup

### 4.1 Test Import Structure

Create `test_imports.py`:

```python
#!/usr/bin/env python3
"""Test script to verify all imports work correctly."""

def test_imports():
    try:
        import fastapi, uvicorn, pydantic
        print("✅ FastAPI stack imported successfully")

        from src.config.settings import get_settings
        from src.utils.logger import get_logger
        from src.exceptions.base_exceptions import ChatServiceException
        print("✅ Our modules imported successfully")

        import motor, redis, asyncpg
        print("✅ Database drivers imported successfully")

        import pytest, black, mypy
        print("✅ Development tools imported successfully")

        print("\n🎉 All dependencies installed correctly!")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

    return True

if __name__ == "__main__":
    test_imports()
```

Run it:

```bash
python test_imports.py
```

### 4.2 Test Application Startup

```bash
python -c "from src.main import app; print('✅ FastAPI app created successfully')"
python -c "from src.config.settings import get_settings; s=get_settings(); print(f'✅ Environment: {s.ENVIRONMENT}')"
```

---

## Common Issues and Solutions

### Issue 1: Module Not Found

```bash
pip install -e .
# or
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### Issue 2: Version Conflicts

```bash
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements-dev.txt
```

### Issue 3: Pre-commit Hook Errors

```bash
pre-commit install
pre-commit autoupdate
pre-commit run --all-files
```

---

## Final Verification Steps

### 5.1 Create Run Configuration

* **Run → Edit Configurations → + → Python**

  * Script path: `src/main.py`
  * Working directory: Project root
  * Env vars: load from `.env`

### 5.2 Test Run

Click ▶️ Run button → Should see startup logs:

```bash
INFO - Chat Service starting version=2.0.0 environment=development
INFO - Chat Service startup completed successfully
```

### 5.3 Test Health Endpoint

```bash
curl http://localhost:8001/health
```

Expected:

```json
{
  "status": "healthy",
  "service": "chat-service",
  "version": "2.0.0"
}
```

---

## Logging Fix Summary

* Convert all enum values using `.value`:

  * `settings.LOG_LEVEL.value`
  * `settings.ENVIRONMENT.value`
* Removed custom `log_config` from uvicorn settings
* Use `uvicorn.run(**uvicorn_config)` with string-based config values

---

## Quick Setup Summary

```bash
# 1. Create project and navigate to directory
cd chat-service

# 2. Run automated setup
chmod +x scripts/setup.sh
./scripts/setup.sh

# 3. Activate environment in PyCharm (Python Interpreter → ./venv)

# 4. Test the setup
python test_imports.py
```

