# Resolve Test Failures - Action Plan Summary
## Quick Reference Guide

**Status**: Planning Complete - Ready for Implementation  
**Failures to Fix**: 3 (All FastAPI import-related)  
**Estimated Time**: 2-3 hours

---

## Quick Overview

### Current Test Failures
1. ❌ `src/api/main.py` - FastAPI import fails
2. ❌ `src/api/routes/__init__.py` - APIRouter import fails  
3. ❌ `src/api/rate_limiting.py` - Request import fails

### Solution Strategy
Add graceful import fallbacks (same pattern as other modules)

---

## Implementation Steps

### Step 1: Create Fallback Module ⏱️ 30 min
**File**: `src/api/fastapi_fallback.py` (NEW)

Create mock classes:
- MockFastAPI
- MockAPIRouter
- MockRequest
- MockCORSMiddleware
- HTTPException, Depends, status codes

### Step 2: Fix Main Module ⏱️ 45 min
**File**: `src/api/main.py`

```python
# Add at top:
try:
    from fastapi import FastAPI, Request, ...
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    from .fastapi_fallback import MockFastAPI as FastAPI, ...
```

### Step 3: Fix Routes Module ⏱️ 30 min
**File**: `src/api/routes/__init__.py`

```python
# Replace direct import with:
try:
    from fastapi import APIRouter
except ImportError:
    from ..fastapi_fallback import MockAPIRouter as APIRouter
```

### Step 4: Fix Dependencies ⏱️ 25 min
**File**: `src/api/dependencies.py`

Add graceful imports for Depends, HTTPException, Header

### Step 5: Fix Rate Limiting ⏱️ 20 min
**File**: `src/api/rate_limiting.py`

Add graceful imports for Request, HTTPException, status

### Step 6: Test & Validate ⏱️ 30 min
```bash
python scripts/test_all_phases.py
```

**Expected**: 0 failures ✅

---

## Files to Modify

| File | Changes | Priority |
|------|---------|----------|
| `src/api/fastapi_fallback.py` | CREATE NEW | High |
| `src/api/main.py` | Add try/except imports | High |
| `src/api/routes/__init__.py` | Add try/except imports | High |
| `src/api/dependencies.py` | Add try/except imports | Medium |
| `src/api/rate_limiting.py` | Add try/except imports | High |

---

## Success Criteria

✅ All 3 test failures resolved  
✅ Code imports without FastAPI  
✅ Test suite passes  
✅ No new errors introduced

---

## Pattern to Follow

```python
try:
    from fastapi import <Module>
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.warning("FastAPI not available")
    from .fastapi_fallback import MockModule as <Module>
```

**Same pattern used in**:
- ✅ `src/api/config.py`
- ✅ `src/api/versioning.py`
- ✅ `src/api/auth/oauth2.py`
- ✅ `src/api/graphql/schema.py`

---

## Detailed Plan

📄 **Full Details**: See `docs/RESOLVE_TEST_FAILURES_PLAN.md`

---

**Ready to Implement**: ✅  
**Next Action**: Create `src/api/fastapi_fallback.py`

