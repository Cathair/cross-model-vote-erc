"""MARC ERC v3.6.1: 异构三 Agent + EAA 双调用 + 动态 evidence 讨论 + 规则融合.

v3.6.1 相对 v3.6 的改动：
- 删除 CONFUSION_PAIRS（数据集特殊处理 + 标签特殊处理）
- P2 arousal filter 改为通用规则（不依赖 strategy、不依赖固定标签集）
"""

from .config_v3 import ARCHITECTURE_VERSION
from .marc_v3 import MARC_ERC_V3

__all__ = ["MARC_ERC_V3", "ARCHITECTURE_VERSION"]
