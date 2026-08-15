from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from utils import parse_json_response, format_context
import config


ZS_PROMPT = """Analyze the emotion of the last utterance in the following dialogue.
Dialogue History:
{context}

Target Utterance ({speaker}): {utterance}

Choose the most appropriate label from {label_list}.
Output ONLY the label text, without any additional words or punctuation."""

SA_PROMPT = """You are an expert in emotion analysis. Please comprehensively consider the speaker's intent, contextual coherence, and possible listener reactions to analyze the emotion of the target utterance.

Dialogue History:
{context}

Target Utterance ({speaker}): {utterance}

Choose a label from {label_list}.
Please output your analysis in JSON format ONLY:
{{"emotion": "label", "reason": "brief reason"}}
"""



class BaselineModel:
    def __init__(self, mode="zero_shot", model: str = None, temperature: float = None):
        self.mode = mode
        self.model = model or config.AGENT_MODEL
        self.temperature = 0.0 if temperature is None else temperature
        self.llm = ChatOpenAI(model=self.model, temperature=self.temperature)

    def predict(self, utterance, speaker, context_records, label_list=None):
        context_str = format_context(context_records)
        prompt_template = ZS_PROMPT if self.mode == "zero_shot" else SA_PROMPT

        fmt_kwargs = dict(context=context_str, utterance=utterance, speaker=speaker)
        if label_list is not None:
            fmt_kwargs["label_list"] = label_list

        prompt = PromptTemplate.from_template(prompt_template).format(**fmt_kwargs)
        response = self.llm.invoke(prompt)

        if self.mode == "zero_shot":
            pred = response.content.strip().lower()
            if label_list:
                from utils import validate_label
                pred = validate_label(pred, label_list)
            return pred, {}
        else:
            parsed = parse_json_response(response.content)
            if parsed:
                emotion = parsed.get("emotion", "neutral").lower()
                if label_list:
                    from utils import validate_label
                    emotion = validate_label(emotion, label_list)
                return emotion, parsed
            return "neutral", {}
