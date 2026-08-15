"""Project configuration. API credentials are loaded from environment / .env only."""
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def repo_root() -> Path:
    return _REPO_ROOT


def load_dotenv() -> None:
    """Load .env from repository root if python-dotenv is available."""
    env_path = _REPO_ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv as _load

        _load(env_path, override=False)
    except ImportError:
        pass


load_dotenv()

# Credentials must be supplied via environment (see .env.example).
if os.environ.get("OPENAI_API_BASE") is None:
    os.environ.setdefault("OPENAI_API_BASE", "https://api.openai.com/v1")

# ==================== Model defaults ====================
AGENT_MODEL = "gpt-4o-mini"
ARBITER_MODEL = "gpt-4o-mini"
TEMPERATURE = 0.0
MAX_RETRIES = 3

# ==================== Debate defaults (legacy) ====================
MAX_DEBATE_ROUNDS = 2
DEBATE_CONFIDENCE_THRESHOLD = 0.6

# ==================== Dataset ====================
DATA_ROOT = str(_REPO_ROOT / "data")

DATASET_CONFIGS = {
    "iemocap": {
        "data_dir": os.path.join(DATA_ROOT, "IEMOCAP"),
        "file_pattern": "{split}.raw.json",
        "format": "json_lines",
    },
    "meld": {
        "data_dir": os.path.join(DATA_ROOT, "MELD"),
        "file_pattern": "{split}.json",
        "format": "json",
    },
}

LABEL_STRATEGIES = {
    "iemocap_6class": {
        "happy": "happy",
        "excited": "excited",
        "sad": "sad",
        "angry": "angry",
        "neutral": "neutral",
        "frustrated": "frustrated",
    },
    "meld_raw": {
        "neutral": "neutral",
        "joy": "joy",
        "sadness": "sadness",
        "anger": "anger",
        "fear": "fear",
        "disgust": "disgust",
        "surprise": "surprise",
    },
}

DATASET_DEFAULT_STRATEGY = {
    "iemocap": "iemocap_6class",
    "meld": "meld_raw",
}


def get_label_list(strategy_name: str) -> list:
    mapping = LABEL_STRATEGIES[strategy_name]
    return sorted(set(mapping.values()))


def get_label_map(strategy_name: str) -> dict:
    return LABEL_STRATEGIES[strategy_name].copy()
