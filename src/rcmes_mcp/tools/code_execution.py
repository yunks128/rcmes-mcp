"""
Python Code Execution Tool

Provides sandboxed Python code execution for custom climate data analysis.
The executed code has access to scientific libraries and loaded datasets.
"""

from __future__ import annotations

import contextlib
import io
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend

from rcmes_mcp.utils.session import session_manager

# Modules the executed code is allowed to import
_ALLOWED_MODULES = frozenset({
    "xarray", "numpy", "pandas", "matplotlib", "matplotlib.pyplot",
    "scipy", "scipy.stats", "scipy.signal", "scipy.ndimage",
    "math", "datetime", "json", "collections", "itertools", "functools",
    "statistics", "operator", "copy", "re", "textwrap",
})

# Safe subset of builtins
_SAFE_BUILTINS = {
    "print": print,
    "len": len,
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "sorted": sorted,
    "reversed": reversed,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "round": round,
    "any": any,
    "all": all,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "frozenset": frozenset,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "bytes": bytes,
    "complex": complex,
    "type": type,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "hasattr": hasattr,
    "getattr": getattr,
    "setattr": setattr,
    "callable": callable,
    "repr": repr,
    "format": format,
    "hash": hash,
    "id": id,
    "iter": iter,
    "next": next,
    "slice": slice,
    "property": property,
    "staticmethod": staticmethod,
    "classmethod": classmethod,
    "super": super,
    "object": object,
    # Exceptions
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "RuntimeError": RuntimeError,
    "StopIteration": StopIteration,
    "AttributeError": AttributeError,
    "ZeroDivisionError": ZeroDivisionError,
    "Exception": Exception,
    "True": True,
    "False": False,
    "None": None,
}


_real_import = __import__

def _safe_import(name: str, globals=None, locals=None, fromlist=(), level=0):
    """Import function that only allows whitelisted modules."""
    base = name.split(".")[0]
    if base not in {m.split(".")[0] for m in _ALLOWED_MODULES}:
        raise ImportError(f"Import of '{name}' is not allowed. Allowed: {', '.join(sorted(_ALLOWED_MODULES))}")
    return _real_import(name, globals, locals, fromlist, level)


def _get_dataset(dataset_id: str):
    """Helper for user code to access a loaded dataset."""
    return session_manager.get(dataset_id)


def _store_dataset(data, description: str = "custom analysis result", variable: str = "custom"):
    """Helper for user code to store a new dataset."""
    dataset_id = session_manager.store(
        data=data,
        source="code_execution",
        variable=variable,
        description=description,
    )
    return dataset_id


def _get_downloads_dir():
    """Get the downloads directory, creating if needed."""
    import os
    from pathlib import Path
    downloads_dir = Path(os.environ.get("RCMES_DOWNLOADS_DIR", Path.home() / ".rcmes" / "downloads"))
    downloads_dir.mkdir(parents=True, exist_ok=True)
    return downloads_dir


def _save_to_netcdf(data, filepath: str | None = None):
    """Save xarray data to NetCDF and return a download URL.

    Args:
        data: xarray Dataset or DataArray
        filepath: Optional custom path. If None, auto-generates in downloads dir.

    Returns:
        Download URL string
    """
    from pathlib import Path

    import xarray as xr
    if hasattr(data, 'compute'):
        data = data.compute()
    if isinstance(data, xr.DataArray):
        data = data.to_dataset(name=data.name or "data")

    if filepath is None:
        import uuid
        filepath = str(_get_downloads_dir() / f"export_{uuid.uuid4().hex[:8]}.nc")

    outpath = Path(filepath)
    # Save to downloads dir for serving
    downloads_dir = _get_downloads_dir()
    serve_path = downloads_dir / outpath.name
    data.to_netcdf(serve_path, engine='scipy')

    # Register for download
    try:
        from rcmes_mcp.api import _create_download_link
        url = _create_download_link(serve_path, outpath.name, "application/x-netcdf")
        return url
    except Exception:
        return str(serve_path)


def _save_to_csv(data, filepath: str | None = None):
    """Save xarray/pandas data to CSV and return a download URL.

    Args:
        data: xarray Dataset, DataArray, or pandas DataFrame
        filepath: Optional custom path. If None, auto-generates in downloads dir.

    Returns:
        Download URL string
    """
    from pathlib import Path
    if hasattr(data, 'compute'):
        data = data.compute()
    if hasattr(data, 'to_dataframe'):
        df = data.to_dataframe().reset_index()
    else:
        df = data

    if filepath is None:
        import uuid
        filepath = str(_get_downloads_dir() / f"export_{uuid.uuid4().hex[:8]}.csv")

    outpath = Path(filepath)
    downloads_dir = _get_downloads_dir()
    serve_path = downloads_dir / outpath.name
    df.to_csv(serve_path, index=False)

    # Register for download
    try:
        from rcmes_mcp.api import _create_download_link
        url = _create_download_link(serve_path, outpath.name, "text/csv")
        return url
    except Exception:
        return str(serve_path)


def execute_python_code(code: str) -> dict:
    """
    Execute Python code for custom climate data analysis.

    The code runs in a sandboxed environment with access to:
    - xarray (as xr), numpy (as np), pandas (as pd)
    - matplotlib.pyplot (as plt), scipy
    - get_dataset(id) to access loaded datasets
    - store_dataset(data, description) to save results
    - print() to show output

    Args:
        code: Python code to execute

    Returns:
        Dictionary with stdout, stderr, images, and any errors
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import xarray as xr

    # Build the execution namespace
    namespace: dict[str, Any] = {
        "__builtins__": {**_SAFE_BUILTINS, "__import__": _safe_import},
        # Pre-imported libraries
        "np": np,
        "xr": xr,
        "pd": pd,
        "plt": plt,
        "xarray": xr,
        "numpy": np,
        "pandas": pd,
        # Dataset helpers
        "get_dataset": _get_dataset,
        "store_dataset": _store_dataset,
        "save_to_netcdf": _save_to_netcdf,
        "save_to_csv": _save_to_csv,
    }

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    result: dict[str, Any] = {
        "stdout": "",
        "stderr": "",
        "images": [],
        "error": None,
    }

    def _exec_code():
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            exec(compile(code, "<user_code>", "exec"), namespace)

    # Run with timeout
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_exec_code)
            future.result(timeout=120)  # 2 minute timeout
    except FuturesTimeoutError:
        result["error"] = "Code execution timed out (120 second limit)"
        plt.close("all")
        return result
    except Exception as e:
        # Capture the actual error
        error_msg = str(e)
        # Check stderr for more details
        stderr_text = stderr_buf.getvalue()
        if stderr_text:
            error_msg = f"{error_msg}\n{stderr_text}"
        result["error"] = error_msg

    # Capture stdout
    result["stdout"] = stdout_buf.getvalue()[:20000]  # Limit output size
    result["stderr"] = stderr_buf.getvalue()[:5000]

    # Capture any matplotlib figures
    import base64
    fig_nums = plt.get_fignums()
    for fig_num in fig_nums:
        fig = plt.figure(fig_num)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        result["images"].append(base64.b64encode(buf.read()).decode("utf-8"))
        buf.close()
    plt.close("all")

    # Check if store_dataset was called — look for dataset_id in namespace
    # (the user code might have stored the return value)

    return result
