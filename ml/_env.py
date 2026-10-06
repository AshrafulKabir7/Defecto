"""Keep every cache on the project drive (the system drive on the dev laptop is nearly full).

Import this module before torch / anomalib / timm.
"""

import os
from pathlib import Path

_CACHE = Path(__file__).resolve().parents[1] / ".cache"
os.environ.setdefault("HF_HOME", str(_CACHE / "hf"))
os.environ.setdefault("TORCH_HOME", str(_CACHE / "torch"))
os.environ.setdefault("TMP", str(_CACHE / "tmp"))
os.environ.setdefault("TEMP", str(_CACHE / "tmp"))
