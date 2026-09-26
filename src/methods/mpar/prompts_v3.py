"""MPAR Prompts：Phase0 / EAA / 动态 evidence 讨论."""

# Phase 0 — 与 v1 角色一致，不含混淆对 Rubric
SVA_PROMPT_V3 = """You are an expert in Speaker Viewpoint sentiment analysis. When analyzing the target utterance:
1. Focus on the speaker's intent and surface expression.
2. Extract explicit emotional words, punctuation, and rhetorical devices.
3. Assess emotional intensity (1-5).
4. Provide emotion prediction and confidence (0.0-1.0).

The emotion label MUST be chosen from: {label_list}

Dialogue History:
{context}

Target Utterance (Speaker: {speaker}): {utterance}

Output JSON ONLY:
{{"emotion": "label", "intensity": 1-5, "confidence": 0.0-1.0, "evidence_span": "exact words/phrases from context or utterance", "reasoning": "one sentence"}}
"""

CIA_PROMPT_V3 = """You are an expert in Contextual Inference. Analyze the target utterance in dialogue context:
1. Detect emotional turning points and coherence with prior turns.
2. Note implicit cues (suppression, shift) — do NOT finalize sarcasm alone; flag cues only.
3. Evaluate contextual emotional coherence (1-5).

The emotion label MUST be chosen from: {label_list}

Dialogue History:
{context}

Target Utterance (Speaker: {speaker}): {utterance}

Output JSON ONLY:
{{"emotion": "label", "emotion_shift": "No shift/Significant shift/Progression", "implicit_cue": "brief note or none", "coherence_score": 1-5, "confidence": 0.0-1.0, "evidence_span": "exact words/phrases from context or utterance", "reasoning": "one sentence"}}
"""

LRA_PROMPT_V3 = """You are an expert in Listener Reaction analysis. Based on dialogue and target utterance:
1. Infer the speaker's emotion via social/listener perspective.
2. Consider power dynamics and interaction impact.
3. Predict how a listener might react emotionally.

The emotion label MUST be chosen from: {label_list}

Dialogue History:
{context}

Target Utterance (Speaker: {speaker}): {utterance}

Output JSON ONLY:
{{"inferred_speaker_emotion": "label", "predicted_listener_emotion": "label", "influence_type": "Positive/Negative/None", "confidence": 0.0-1.0, "evidence_span": "exact words/phrases from context or utterance", "reasoning": "one sentence"}}
"""

PHASE0_PROMPTS = {
    "SVA": SVA_PROMPT_V3,
    "CIA": CIA_PROMPT_V3,
    "LRA": LRA_PROMPT_V3,
}

# ---------- EAA Call-1: 证据质量（不看反讽、不看融合标签）----------
EAA_SCORE_PROMPT = """You are an Evidence Quality Assessor (NOT an emotion classifier).

Evaluate each expert's evidence ONLY. Do NOT judge sarcasm or final emotion labels.

Dialogue History:
{context}

Target Utterance (Speaker: {speaker}): {utterance}

Expert outputs:
{experts_block}

For each expert (SVA, CIA, LRA), score 0.0-1.0:
- evidence_grounded: quoted span actually appears in context/utterance
- label_specificity: evidence supports THIS label, not multiple confused labels
- context_aligned: reasoning fits dialogue flow

Optionally note which two labels seem most confusable (soft hint only, NOT a final label):
- ambiguous_between: null OR [label_a, label_b] from the expert's stated label and the most plausible alternative

Output JSON ONLY:
{{"SVA": {{"evidence_grounded": 0.0, "label_specificity": 0.0, "context_aligned": 0.0, "ambiguous_between": null}},
  "CIA": {{"evidence_grounded": 0.0, "label_specificity": 0.0, "context_aligned": 0.0, "ambiguous_between": null}},
  "LRA": {{"evidence_grounded": 0.0, "label_specificity": 0.0, "context_aligned": 0.0, "ambiguous_between": null}}}}
"""

# ---------- EAA Call-2: 语用/反讽（不输入各 Agent label）----------
EAA_PRAG_PROMPT = """You are a Pragmatics Analyst (NOT an emotion classifier).

Analyze surface vs intended sentiment and sarcasm likelihood.
Do NOT output a final emotion label from the dataset taxonomy.

Dialogue History:
{context}

Target Utterance (Speaker: {speaker}): {utterance}

Output JSON ONLY:
{{"sarcasm_likelihood": 0.0,
  "surface_sentiment": "positive|neutral|negative",
  "intended_sentiment": "positive|neutral|negative",
  "mismatch": false,
  "cue_types": ["context_contradiction", "mock_praise", "exaggeration"]}}
"""

# ---------- Phase 2: 讨论 prompt（2:1 blind / 1:1:1 split）----------
DISCUSSION_PROMPT_SPLIT = """You are {agent_name}, re-evaluating your emotion judgment after structured review.

Dialogue History:
{context}

Target Utterance (Speaker: {speaker}): {utterance}

Your initial label: {initial_label}
Your initial evidence: {initial_evidence}

Other experts (labels + evidence only):
{others_block}

{discussion_constraints}

Rules:
1. Output choice among allowed labels ONLY.
2. key_cue must cite dialogue (one sentence).
3. confidence_ordinal: high|medium|low (your certainty after review).

Output JSON ONLY:
{{"choice": "label", "revised_from_initial": true/false, "key_cue": "...", "confidence_ordinal": "high|medium|low"}}
"""

DISCUSSION_PROMPT_2TO1 = """You are {agent_name}, re-evaluating your emotion judgment after structured review.

Dialogue History:
{context}

Target Utterance (Speaker: {speaker}): {utterance}

Two candidate interpretations (each label with its strongest supporting evidence):
{options_block}

{discussion_constraints}

Rules:
1. Output choice among allowed labels ONLY — compare evidence grounding and specificity only.
2. Do NOT infer vote counts or consensus; neither option is "majority" by default.
3. key_cue must cite dialogue (one sentence).
4. confidence_ordinal: high|medium|low (your certainty after review).

Output JSON ONLY:
{{"choice": "label", "revised_from_initial": true/false, "key_cue": "...", "confidence_ordinal": "high|medium|low"}}
"""

# 兼容旧名
DISCUSSION_PROMPT = DISCUSSION_PROMPT_SPLIT

DISCUSSION_DYNAMIC_CONSTRAINT_SPLIT = """
DISCUSSION MODE — Evidence comparison (no dataset-specific cheat sheet).

Candidate labels (from expert disagreement only): {candidate_labels}
You MUST choose ONE label from this list only.

Review task:
1. Compare each expert's evidence_span — which label is most specifically supported?
2. If your initial label differs from BOTH others (you are the sole dissenting vote), change ONLY if their evidence clearly outweighs yours on grounding and specificity.
3. Do not switch merely because two others agree — majority is not evidence.

{prag_block}

Counterfactual template: If label X were wrong, what concrete cue in the dialogue would rule it out?
"""

DISCUSSION_DYNAMIC_CONSTRAINT_2TO1 = """
DISCUSSION MODE — Blind evidence comparison (no self-initial, no vote counts).

Candidate labels: {candidate_labels}
You MUST choose ONE label from this list only.

Review task:
1. Compare the two options solely on evidence grounding and label specificity in the dialogue.
2. Choose the label best supported by concrete cues — listing order carries no meaning.

{prag_block}

Counterfactual template: If label X were wrong, what concrete cue in the dialogue would rule it out?
"""

DISCUSSION_DYNAMIC_CONSTRAINT = DISCUSSION_DYNAMIC_CONSTRAINT_SPLIT

ALL_PROMPTS_V3 = {
    "phase0": PHASE0_PROMPTS,
    "eaa_score": EAA_SCORE_PROMPT,
    "eaa_prag": EAA_PRAG_PROMPT,
    "discussion_split": DISCUSSION_PROMPT_SPLIT,
    "discussion_2to1": DISCUSSION_PROMPT_2TO1,
    "discussion_dynamic_split": DISCUSSION_DYNAMIC_CONSTRAINT_SPLIT,
    "discussion_dynamic_2to1": DISCUSSION_DYNAMIC_CONSTRAINT_2TO1,
}
