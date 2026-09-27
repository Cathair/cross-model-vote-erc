"""MPAR model slots, role priors, thresholds, and EAA call policy."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import MAX_RETRIES  # noqa: E402

ARCHITECTURE_VERSION = "mpar"

V3_ALLOWED_MODELS = (
    "gpt-4o",
    "gemini-2.5-flash-lite",
    "claude-3-haiku-20240307",
    "qwen-plus",
)

_V3_ALL_MODEL = os.environ.get("MARC_V3_ALL_MODEL")
if _V3_ALL_MODEL:
    EAA_MODEL = _V3_ALL_MODEL
    BASELINE_MODEL = _V3_ALL_MODEL
    AGENT_MODELS = {"SVA": _V3_ALL_MODEL, "CIA": _V3_ALL_MODEL, "LRA": _V3_ALL_MODEL}
else:
    EAA_MODEL = "gemini-2.5-flash-lite"
    BASELINE_MODEL = EAA_MODEL
    AGENT_MODELS = {
        "SVA": "claude-3-haiku-20240307",
        "CIA": "qwen-plus",
        "LRA": "gpt-4o",
    }

AGENT_TEMP = 0.0

BASE_ROLE_WEIGHT = {
    "SVA": 0.35,
    "CIA": 0.40,
    "LRA": 0.25,
}

PRAG_SVA_FACTOR = 0.5
PRAG_CIA_FACTOR = 1.3
PRAG_LRA_FACTOR = 1.0

TAU_S = 0.7
TAU_SSPEC = float(os.environ.get("MARC_TAU_SSPEC", "0.40"))
TAU_G = float(os.environ.get("MARC_TAU_G", "0.65"))
P2_AROUSAL_GATE = os.environ.get("MARC_P2_AROUSAL", "1") == "1"

SKIP_DEEP = os.environ.get("MARC_SKIP_DEEP", os.environ.get("MARC_IEM_NO_DEEP", "0")) == "1"
DEEP_MODE = os.environ.get("MARC_DEEP_MODE", "post21_cia")
CIA_FALLBACK = os.environ.get("MARC_CIA_FALLBACK", "1") == "1"
CIA_FALLBACK_MARGIN = float(os.environ.get("MARC_CIA_FALLBACK_MARGIN", "0.05"))
ORDINAL_MAP = {"high": 1.0, "medium": 0.6, "low": 0.3}
FUSION_MARGIN_EPSILON = 0.05

FAST_ONLY = os.environ.get("MARC_FAST_ONLY", "0") == "1"
FAST_FIRST = os.environ.get("MARC_FAST_FIRST", "0") == "1"
TAU_MARGIN = float(os.environ.get("MARC_TAU_MARGIN", "0.10"))
ARBITRATOR_AGENT = os.environ.get("MARC_ARBITRATOR", "CIA")
NO_EAA_PRAG = os.environ.get("MARC_NO_EAA_PRAG", "0") == "1"

CONFUSION_PAIRS = {}

AGENT_NAMES = ("SVA", "CIA", "LRA")
