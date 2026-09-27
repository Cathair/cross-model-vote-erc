import json
import os
from typing import List, Dict, Optional
from config import DATASET_CONFIGS, DATASET_DEFAULT_STRATEGY, get_label_map


class DialogueSample:
    def __init__(self, utterance, speaker, label, context, dialogue_id, utterance_id):
        self.utterance = utterance
        self.speaker = speaker
        self.label = label
        self.context = context
        self.dialogue_id = dialogue_id
        self.utterance_id = utterance_id

    def __repr__(self):
        return (f"DialogueSample(dlg={self.dialogue_id}, utt_id={self.utterance_id}, "
                f"speaker={self.speaker}, label={self.label})")


def _parse_iemocap_file(filepath: str) -> List[List[Dict]]:
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    try:
        data = json.loads(content)
        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict) and "text" in data[0]:
                return [data]
            elif len(data) > 0 and isinstance(data[0], list):
                return data
            else:
                return [data]
    except json.JSONDecodeError:
        pass

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
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, dict):
        return data

    if isinstance(data, list):
        groups = {}
        for item in data:
            dlg_id = str(item.get("Dialogue_ID", item.get("dialogue_id", "unknown")))
            if dlg_id not in groups:
                groups[dlg_id] = []
            utt = {
                "text": item.get("Utterance", item.get("utterance", item.get("text", ""))),
                "speaker": item.get("Speaker", item.get("speaker", "Unknown")),
                "label": item.get("Emotion", item.get("emotion", item.get("label", "neutral"))),
                "utterance_id": item.get("Utterance_ID", item.get("utterance_id", 0)),
            }
            groups[dlg_id].append(utt)
        for dlg_id in groups:
            groups[dlg_id].sort(key=lambda x: x.get("utterance_id", 0))
        return groups

    raise ValueError(f"Unknown MELD format in {filepath}")


def _apply_label_map(raw_label: str, label_map: dict,
                     strategy: str, filter_unmapped: bool = True) -> Optional[str]:
    raw_label = raw_label.lower().strip()
    if raw_label in label_map:
        return label_map[raw_label]

    for key in label_map:
        if raw_label.startswith(key) or key.startswith(raw_label):
            return label_map[key]

    if filter_unmapped:
        return None
    return "neutral"


def load_iemocap(filepath: str, strategy: str = "meld_aligned",
                 filter_unmapped: bool = True) -> List[DialogueSample]:
    label_map = get_label_map(strategy)
    dialogues = _parse_iemocap_file(filepath)

    samples = []
    for dlg_idx, dialogue in enumerate(dialogues):
        dialogue_id = f"iemocap_dlg_{dlg_idx}"
        context = []

        for utt_idx, utt in enumerate(dialogue):
            raw_label = utt.get("label", utt.get("emotion", "neutral"))
            mapped_label = _apply_label_map(raw_label, label_map, strategy, filter_unmapped)

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


def load_meld(filepath: str, strategy: str = "meld_raw") -> List[DialogueSample]:
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


def print_label_distribution(samples: List[DialogueSample]):
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
