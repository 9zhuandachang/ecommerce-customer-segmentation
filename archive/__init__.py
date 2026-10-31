"""Earlier experiments and their preserved artifacts."""

import sys
from types import ModuleType
from src.data import transforms

_legacy = sys.modules.setdefault("stage4", ModuleType("stage4"))
_legacy.__path__ = getattr(_legacy, "__path__", [])
_legacy.transforms = transforms
sys.modules.setdefault("stage4.transforms", transforms)
