"""InsideOut (Mozikov et al., ECAI 2024) ERC prompts.

Architecture: 5 Ekman emotion-perspective agents + 1 Aggregate Agent (star topology).
Reimplemented for IEMOCAP/MELD following CMTD (ACL Findings 2026) description;
CMTD cites Table 6 for prompts but does not ship inside_out.txt in the public repo.

Each emotion agent analyzes dialogue from one Ekman perspective and outputs
emotion + confidence + rationale (JSON). The Aggregate Agent synthesizes
all five analyses into a final label from the dataset label set.
"""

EKMAN_EMOTIONS = ("anger", "disgust", "fear", "happiness", "sadness")

EMOTION_AGENT_PROMPT = """You are an emotional agent in the InsideOut multi-agent emotion recognition framework.
You specialize in the emotion of {emotion_perspective} (Paul Ekman's basic emotion taxonomy).
Analyze the dialogue from the perspective of {emotion_perspective}: you are especially attuned to
verbal and contextual cues associated with {emotion_perspective}, while still considering the full context.

Dialogue History:
{context}

Target Utterance ({speaker}): {utterance}

Task: Determine the primary emotion expressed in the TARGET utterance.
Choose exactly ONE label from: {label_list}

Output JSON ONLY (no markdown, no extra text):
{{"emotion": "<label>", "confidence": <number between 0 and 1>, "rationale": "<brief reason>"}}"""

AGGREGATE_AGENT_PROMPT = """You are the Aggregate Agent in the InsideOut multi-agent emotion recognition framework.
Five specialized emotional agents (anger, disgust, fear, happiness, sadness) have independently analyzed
the target utterance from their respective emotional perspectives.

Dialogue History:
{context}

Target Utterance ({speaker}): {utterance}

Agent Analyses:
{agent_analyses}

Task: Synthesize all agent perspectives and determine the primary emotion of the TARGET utterance.
Choose exactly ONE label from: {label_list}

Output JSON ONLY (no markdown, no extra text):
{{"emotion": "<label>", "rationale": "<brief synthesis of agent perspectives>"}}"""
