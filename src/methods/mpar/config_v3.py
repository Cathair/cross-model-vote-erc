"""MPAR 配置：模型槽位、先验、阈值、EAA 调用策略."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import get_label_list, DATASET_DEFAULT_STRATEGY, TEMPERATURE, MAX_RETRIES  # noqa: E402

ARCHITECTURE_VERSION = "mpar"

# ---------- 模型分配（仅限等价性报告 §二 四模型，一模型一槽位）----------
# 四模型：gpt-4o · gemini-2.5-flash-lite · claude-3-haiku-20240307 · qwen-plus
# 约束：MPAR pipeline 不得使用上述四模型以外的 API ID（含 LRA）。
# 统计：12 对 Acc + 12 对 W-F1 全部 ns（等价性验证，见论文补充材料）

V3_ALLOWED_MODELS = (
    "gpt-4o",
    "gemini-2.5-flash-lite",
    "claude-3-haiku-20240307",
    "qwen-plus",
)

# Meta 层 + ZeroShot / SingleAgent 对照（与 EAA 同 backbone）
# MARC_V3_ALL_MODEL：四槽位统一模型（如 all-qwen 消融实验）
_V3_ALL_MODEL = os.environ.get("MARC_V3_ALL_MODEL")
if _V3_ALL_MODEL:
    EAA_MODEL = _V3_ALL_MODEL
    BASELINE_MODEL = _V3_ALL_MODEL
    AGENT_MODELS = {"SVA": _V3_ALL_MODEL, "CIA": _V3_ALL_MODEL, "LRA": _V3_ALL_MODEL}
else:
    EAA_MODEL = "gemini-2.5-flash-lite"       # IEMOCAP Acc 0.6333（并列最高）；MELD Acc 0.6133
    BASELINE_MODEL = EAA_MODEL
    # 三分类 Agent（四模型中剩余三槽；两数据集固定）
    AGENT_MODELS = {
        "SVA": "claude-3-haiku-20240307",     # IEMOCAP W-F1 最高 0.6139；Acc 0.6333 并列最高
        "CIA": "qwen-plus",                   # MELD Acc/W-F1 最高 0.6267 / 0.6365
        "LRA": "gpt-4o",                      # 四模型末槽；IEMOCAP Acc 0.6083 / MELD 0.6067，两集最均衡
    }

AGENT_TEMP = 0.0

# ---------- 角色先验（IEMOCAP 固定表，两数据集统一）----------
BASE_ROLE_WEIGHT = {
    "SVA": 0.35,
    "CIA": 0.40,
    "LRA": 0.25,
}

# 反讽触发的 base 修正系数
PRAG_SVA_FACTOR = 0.5
PRAG_CIA_FACTOR = 1.3
PRAG_LRA_FACTOR = 1.0

# ---------- 阈值 ----------
TAU_S = 0.7           # sarcasm_likelihood 触发讨论 / base 修正
TAU_SSPEC = float(os.environ.get("MARC_TAU_SSPEC", "0.40"))      # label_specificity 低于此视为证据不专属 → 可触发讨论（§六 离线优化：0.6→0.40；可用 MARC_TAU_SSPEC 覆盖）
TAU_G = float(os.environ.get("MARC_TAU_G", "0.65"))
# P2：高唤醒 utterance 下抑制 neutral 等（IEM + MELD 统一规则）
P2_AROUSAL_GATE = os.environ.get("MARC_P2_AROUSAL", "1") == "1"

# 跳过 Deep 讨论（两数据集相同，MARC_SKIP_DEEP=1；MARC_IEM_NO_DEEP 兼容旧名）
SKIP_DEEP = os.environ.get("MARC_SKIP_DEEP", os.environ.get("MARC_IEM_NO_DEEP", "0")) == "1"
# Deep 路径决策（两数据集相同）：fusion | post21_minority | post21_cia
# 默认 post21_cia：Deep 讨论后 2:1 投票时采用 CIA 标签
DEEP_MODE = os.environ.get("MARC_DEEP_MODE", "post21_cia")
# Fast 融合 margin 低于阈值时回退 CIA（默认开，margin=0.05）
CIA_FALLBACK = os.environ.get("MARC_CIA_FALLBACK", "1") == "1"
CIA_FALLBACK_MARGIN = float(os.environ.get("MARC_CIA_FALLBACK_MARGIN", "0.05"))
ORDINAL_MAP = {"high": 1.0, "medium": 0.6, "low": 0.3}
FUSION_MARGIN_EPSILON = 0.05  # 分析层：top1-top2 低于此视为低 margin（不改预测）

# 方案 B：跳过所有 Deep 讨论，Phase-0 + EAA + Fast 融合（MARC_FAST_ONLY=1）
FAST_ONLY = os.environ.get("MARC_FAST_ONLY", "0") == "1"

# 方案 A：两阶段仲裁（MARC_FAST_FIRST=1）
# Stage1：Phase-0 Fast 融合；margin >= TAU_MARGIN → 直接输出
# Stage2：低 margin + 原路由会进 Deep → 单 Agent 仲裁（默认 CIA），非三 Agent 讨论
FAST_FIRST = os.environ.get("MARC_FAST_FIRST", "0") == "1"
TAU_MARGIN = float(os.environ.get("MARC_TAU_MARGIN", "0.10"))
ARBITRATOR_AGENT = os.environ.get("MARC_ARBITRATOR", "CIA")

# 消融：完全跳过 EAA Call-2（MARC_NO_EAA_PRAG=1）
NO_EAA_PRAG = os.environ.get("MARC_NO_EAA_PRAG", "0") == "1"

# ---------- EAA Call-2 调用策略（两数据集相同）----------
# Call-1 (EAA-Score): 始终执行
# Call-2 (EAA-Prag): 三 Agent Phase-0 标签完全一致 → skip；否则执行（NO_EAA_PRAG 时始终 skip）

# ---------- 混淆对表 ----------
# CONFUSION_PAIRS 已移除；讨论路由依赖 Phase-0 动态候选 + evidence 比较。
CONFUSION_PAIRS = {}

AGENT_NAMES = ("SVA", "CIA", "LRA")
