import re
import os
from typing import Dict

class PolicyTopicClassifier:
    POLICY_TOPICS: Dict[str, str] = {
        "Trade_Tariff": "Statements about tariffs, trade wars, duties, import/export restrictions, China trade negotiations.",
        "Rate_Fed": "Statements about interest rates, inflation, the Federal Reserve, monetary policy, rate hikes or cuts.",
        "China_Conflict": "Statements about China as geopolitical rival, sanctions, accusations, military or diplomatic conflict.",
        "Tech_Regulation": "Statements about Big Tech (Google, Apple, Meta, Amazon), antitrust, censorship, privacy or regulation.",
        "Defense_NationalSecurity": "Statements about national security, NATO, military actions, defense spending, wars or geopolitical risk.",
    }

    GPT_MODEL = "gpt-4.1-mini"

    def __init__(self, mode="gpt_zero_shot"):
        self.mode = mode
        self.client = None
        if mode == "gpt_zero_shot":
            try:
                from openai import OpenAI
                if not os.environ.get("OPENAI_API_KEY"):
                    raise RuntimeError("no key")
                self.client = OpenAI()
            except Exception:
                self.mode = "lexicon_rule"
        self.lexicon = self._build_lexicon()

    def classify(self, text: str) -> str:
        t = (text or "").strip()
        if self.mode != "gpt_zero_shot" or not self.client:
            return self._lexicon_classify(t)
        try:
            prompt = self._build_prompt(t)
            resp = self.client.chat.completions.create(
                model=self.GPT_MODEL,
                temperature=0,
                messages=[
                    {"role": "system", "content": "Classify the policy topic."},
                    {"role": "user", "content": prompt},
                ],
            )
            label = resp.choices[0].message.content.strip()
            return label if label in self.POLICY_TOPICS else "Other"
        except Exception:
            return self._lexicon_classify(t)

    def _build_prompt(self, text: str) -> str:
        topics_formatted = "\n".join([f"- {k}: {v}" for k, v in self.POLICY_TOPICS.items()])
        labels = ", ".join(list(self.POLICY_TOPICS.keys()))
        return (
            f"Classify the following Trump tweet into one of the policy topics below.\n\n"
            f"Policy topics:\n{topics_formatted}\n\n"
            f"Tweet:\n{text}\n\n"
            f"Return only ONE label from this list:\n{labels}\n\n"
            f"If the tweet carries no policy meaning, return: Other"
        )

    def _build_lexicon(self):
        return {
            "Trade_Tariff": [r"tariff", r"trade", r"import", r"export", r"wto", r"duty"],
            "Rate_Fed": [r"fed", r"interest", r"inflation", r"rate hike", r"rate cut", r"powell"],
            "China_Conflict": [r"china", r"ccp", r"sanction", r"virus"],
            "Tech_Regulation": [r"google", r"facebook", r"amazon", r"apple", r"censorship", r"privacy"],
            "Defense_NationalSecurity": [r"military", r"nato", r"defense", r"security", r"missile", r"war"],
        }

    def _lexicon_classify(self, text: str) -> str:
        t = text.lower()
        for topic, patterns in self.lexicon.items():
            for p in patterns:
                if re.search(p, t):
                    return topic
        return "Other"

