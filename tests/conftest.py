"""
Put `examples/` on sys.path for tests that import the runnable scripts directly.

Those scripts are entry points, not a package, so `from run_streaming_evaluation
import ...` only resolves once their directory is importable. Without this the
affected test errors at collection with ModuleNotFoundError rather than failing
on anything it actually checks.
"""
import sys
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))
