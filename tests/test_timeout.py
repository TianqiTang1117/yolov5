import platform
import sys
import types
from pathlib import Path

import numpy as np
import pytest

# Stub minimal cv2 module to satisfy utils.general import without requiring full OpenCV
cv2_stub = types.SimpleNamespace(
    setNumThreads=lambda x: None,
    imshow=lambda *a, **k: None,
    waitKey=lambda *a, **k: None,
    destroyAllWindows=lambda *a, **k: None,
    resize=lambda img, dsize: img,
    imdecode=lambda *a, **k: np.zeros((1, 1, 3), dtype=np.uint8),
    imencode=lambda *a, **k: (None, types.SimpleNamespace(tofile=lambda f: None)),
    imwrite=lambda *a, **k: None,
    imread=lambda *a, **k: None,
    IMREAD_COLOR=1,
)
sys.modules.setdefault("cv2", cv2_stub)

# Stub minimal ultralytics package to satisfy utils.general imports
ultralytics_stub = types.ModuleType("ultralytics")
ultralytics_stub.__version__ = "0.0"
checks_module = types.ModuleType("ultralytics.utils.checks")
checks_module.check_requirements = lambda *a, **k: None
utils_module = types.ModuleType("ultralytics.utils")
utils_module.checks = checks_module
sys.modules["ultralytics"] = ultralytics_stub
sys.modules["ultralytics.utils"] = utils_module
sys.modules["ultralytics.utils.checks"] = checks_module

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.general import Timeout


def test_timeout_suppresses_timeout_error_on_windows(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    with Timeout(1, suppress_timeout_errors=True):
        raise TimeoutError("boom")


def test_timeout_propagates_when_not_suppressed(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    with pytest.raises(TimeoutError):
        with Timeout(1, suppress_timeout_errors=False):
            raise TimeoutError("boom")
