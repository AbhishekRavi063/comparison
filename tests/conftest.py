# Use non-GUI backend for matplotlib in tests (avoids crash in headless/CI)
import os

os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), ".mplconfig"))

# MNE may write config under ~/.mne; force a writable HOME in test context.
_test_home = os.path.join(os.path.dirname(__file__), ".home")
os.makedirs(_test_home, exist_ok=True)
os.environ["HOME"] = _test_home
