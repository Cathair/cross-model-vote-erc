import json
import os
from typing import List, Dict, Optional
from config import (
    DATASET_CONFIGS, LABEL_STRATEGIES, DATASET_DEFAULT_STRATEGY, get_label_map
)


class DialogueSample:
    """单条预测样本（utterance 级别）"""
    def __init__(self, utterance, speaker, label, context, dialogue_id, utterance_id):
        self.utterance = utterance
        self.speaker = speaker
        self.label = label
        self.context = context          # List[Dict]: 前序对话历史
        self.dialogue_id = dialogue_id
        self.utterance_id = utterance_id

    def __repr__(self):
        return (f"DialogueSample(dlg={self.dialogue_id}, utt_id={self.utterance_id}, "
                f"speaker={self.speaker}, label={self.label})")


def _parse_iemocap_file(filepath: str) -> List[List[Dict]]:
    """
    解析 IEMOCAP 文件，返回 List[Dialogue]，每个 Dialogue 是 List[utterance_dict]
    兼容两种格式：
    - JSON Lines: 每行一个对话的 JSON 数组
    - 单个大 JSON 数组: 整个文件就是一个对话
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    # 尝试整体解析为 JSON
    try:
        data = json.loads(content)
        if isinstance(data, list):
            # 检查第一个元素是 utterance dict 还是 dialogue list
            if len(data) > 0 and isinstance(data[0], dict) and "text" in data[0]:
                # 整个文件是一个对话
                return [data]
            elif len(data) > 0 and isinstance(data[0], list):
                # 文件包含多个对话
                return data
            else:
                return [data]
    except json.JSONDecodeError:
        pass

    # JSON Lines 格式：每行一个对话
    dialogues = []
    for line in content.split('\n'):
        line = line.strip()
        if line:
            try:
                dlg = json.loads(line)
                if isinstance(dlg, list):
                    dialogues.append(dlg)
                else:
                    dialogues.append([dlg])
            except json.JSONDecodeError:
                continue
    return dialogues


def _parse_meld_file(filepath: str) -> Dict[str, List[Dict]]:
    """
    解析 MELD 文件，返回 {dialogue_id: [utterance_dict, ...]}
    兼容多种 MELD 格式：
    - {dialogue_id: [{utterance, speaker, emotion}, ...]}
    - [{Dialogue_ID, Utterance_ID, Utterance, Speaker, Emotion}, ...]（扁平CSV转JSON）
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 格式1：按对话 ID 分组的字典
    if isinstance(data, dict):
        return data

    # 格式2：扁平列表，需要按 Dialogue_ID 分组
    if isinstance(data, list):
        groups = {}
        for item in data:
            dlg_id = str(item.get("Dialogue_ID", item.get("dialogue_id", "unknown")))
            if dlg_id not in groups:
                groups[dlg_id] = []
            # 统一字段名
            utt = {
                "text": item.get("Utterance", item.get("utterance", item.get("text", ""))),
                "speaker": item.get("Speaker", item.get("speaker", "Unknown")),
                "label": item.get("Emotion", item.get("emotion", item.get("label", "neutral"))),
                "utterance_id": item.get("Utterance_ID", item.get("utterance_id", 0)),
            }
            groups[dlg_id].append(utt)
        # 按 utterance_id 排序
        for dlg_id in groups:
            groups[dlg_id].sort(key=lambda x: x.get("utterance_id", 0))
        return groups

    raise ValueError(f"Unknown MELD format in {filepath}")


def _apply_label_map(raw_label: str, label_map: dict, 
                     strategy: str, filter_unmapped: bool = True) -> Optional[str]:
    """
    将原始标签映射到策略定义的标准标签
    - filter_unmapped=True: 如果标签不在映射表中，返回 None（跳过该样本）
    - filter_unmapped=False: 未映射的标签归为 "neutral"
    """
    raw_label = raw_label.lower().strip()
    if raw_label in label_map:
        return label_map[raw_label]
    
    # 尝试去除后缀（如 "surprised" -> "surprise"）
    for key in label_map:
        if raw_label.startswith(key) or key.startswith(raw_label):
            return label_map[key]
    
    if filter_unmapped:
        return None
    return "neutral"


def load_iemocap(filepath: str, strategy: str = "meld_aligned",
                 filter_unmapped: bool = True) -> List[DialogueSample]:
    """加载 IEMOCAP 数据集"""
    label_map = get_label_map(strategy)
    dialogues = _parse_iemocap_file(filepath)
    
    samples = []
    for dlg_idx, dialogue in enumerate(dialogues):
        dialogue_id = f"iemocap_dlg_{dlg_idx}"
        context = []
        
        for utt_idx, utt in enumerate(dialogue):
            raw_label = utt.get("label", utt.get("emotion", "neutral"))
            mapped_label = _apply_label_map(raw_label, label_map, strategy, filter_unmapped)
            
            # 跳过无法映射的标签
            if mapped_label is None:
                context.append(utt)  # 仍保留在上下文中
                continue
            
            text = utt.get("text", utt.get("utterance", ""))
            speaker = utt.get("speaker", "Unknown")
            
            samples.append(DialogueSample(
                utterance=text,
                speaker=speaker,
                label=mapped_label,
                context=list(context),  # 深拷贝当前上下文
                dialogue_id=dialogue_id,
                utterance_id=utt_idx,
            ))
            context.append(utt)  # 添加到上下文
    
    return samples


def load_meld(filepath: str, strategy: str = "meld_raw") -> List[DialogueSample]:
    """加载 MELD 数据集"""
    label_map = get_label_map(strategy)
    dialogues = _parse_meld_file(filepath)
    
    samples = []
    for dlg_id, dialogue in dialogues.items():
        dialogue_id = f"meld_{dlg_id}"
        context = []
        
        for utt_idx, utt in enumerate(dialogue):
            raw_label = utt.get("label", utt.get("emotion", "neutral"))
            mapped_label = _apply_label_map(raw_label, label_map, strategy, filter_unmapped=True)
            
            if mapped_label is None:
                context.append(utt)
                continue
            
            text = utt.get("text", utt.get("utterance", ""))
            speaker = utt.get("speaker", "Unknown")
            
            samples.append(DialogueSample(
                utterance=text,
                speaker=speaker,
                label=mapped_label,
                context=list(context),
                dialogue_id=dialogue_id,
                utterance_id=utt_idx,
            ))
            context.append(utt)
    
    return samples


def load_dataset(dataset_name: str, data_dir: str = "data", 
                 split: str = "test", strategy: str = None) -> List[DialogueSample]:
    """
    统一数据加载入口
    
    Args:
        dataset_name: "iemocap" 或 "meld"
        data_dir: 数据根目录
        split: "train", "dev", "test"
        strategy: 标签策略，None 时使用数据集默认策略
    """
    dataset_name = dataset_name.lower()
    config = DATASET_CONFIGS[dataset_name]
    
    if strategy is None:
        strategy = DATASET_DEFAULT_STRATEGY[dataset_name]
    
    filepath = os.path.join(data_dir, config["data_dir"].split("/")[-1], 
                           config["file_pattern"].format(split=split))
    
    print(f"Loading {dataset_name} [{split}] from: {filepath}")
    print(f"Label strategy: {strategy}")
    
    if dataset_name == "iemocap":
        samples = load_iemocap(filepath, strategy)
    elif dataset_name == "meld":
        samples = load_meld(filepath, strategy)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    print(f"Loaded {len(samples)} samples")
    return samples


# ==================== 调试工具 ====================
def print_label_distribution(samples: List[DialogueSample]):
    """打印标签分布"""
    from collections import Counter
    labels = [s.label for s in samples]
    dist = Counter(labels)
    total = len(labels)
    print(f"\n{'='*55}")
    print(f"Total samples: {total}")
    print(f"{'Label':<15} {'Count':<10} {'Ratio':<10}")
    print(f"{'-'*35}")
    for label, count in dist.most_common():
        print(f"{label:<15} {count:<10} {count/total*100:.2f}%")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    # 测试 IEMOCAP 加载（meld_aligned 策略）
    print("="*60)
    print("Test 1: IEMOCAP with 'meld_aligned' strategy")
    print("="*60)
    samples = load_dataset("iemocap", "data", "test", strategy="meld_aligned")
    print_label_distribution(samples)
    
    # 打印前3个样本
    for i, s in enumerate(samples[:3]):
        print(f"\n--- Sample {i+1} ---")
        print(f"  Speaker:    {s.speaker}")
        print(f"  Utterance:  {s.utterance}")
        print(f"  Label:      {s.label}")
        print(f"  Context:    {len(s.context)} turns")

    # 测试 IEMOCAP 加载（iemocap_6class 策略）
    print("\n" + "="*60)
    print("Test 2: IEMOCAP with 'iemocap_6class' strategy")
    print("="*60)
    samples_6c = load_dataset("iemocap", "data", "test", strategy="iemocap_6class")
    print_label_distribution(samples_6c)
