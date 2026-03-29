from evaluator.ml_models.similarity_detector import SIMILARITY_THRESHOLD, get_similarity_signal


INTENT_RULES = [
    {
        "category": "Self Harm",
        "issue": "Self Harm",
        "phrases": ["suicide", "end their life", "end my life", "self-harm"],
        "reason": "Prompt indicates self harm related intent.",
        "severity": 0.95,
    },
    {
        "category": "Violence",
        "issue": "Violence",
        "phrases": ["bomb", "weapon", "hurt another person", "kill", "attack"],
        "reason": "Prompt asks for violent or weapon-related harm.",
        "severity": 0.90,
    },
    {
        "category": "Illegal Activity",
        "issue": "Illegal Activity",
        "phrases": ["illegal drugs", "drug", "fake document", "fake documents", "fraud", "hack"],
        "reason": "Prompt requests illegal or fraudulent activity.",
        "severity": 0.85,
    },
    {
        "category": "Bias / Hate",
        "issue": "Bias / Hate",
        "phrases": ["race superior", "women bad", "immigrants bad", "hate speech"],
        "reason": "Prompt contains discriminatory or hateful intent.",
        "severity": 0.80,
    },
    {
        "category": "Educational",
        "issue": "Educational",
        "phrases": ["explain", "what", "why", "how does", "contributions"],
        "reason": "Prompt appears educational or informational.",
        "severity": 0.15,
    },
]


def classify_prompt_intent(prompt, use_remote_similarity=True):
    prompt_lower = (prompt or "").lower()
    for rule in INTENT_RULES:
        for phrase in rule["phrases"]:
            if phrase in prompt_lower:
                return {
                    "category": rule["category"],
                    "issue": rule["issue"],
                    "matched_phrase": phrase,
                    "reason": rule["reason"],
                    "severity": rule["severity"],
                }

    similarity_signal = get_similarity_signal(prompt, use_remote=use_remote_similarity)
    if similarity_signal["max_similarity"] >= SIMILARITY_THRESHOLD:
        return {
            "category": similarity_signal["issue"],
            "issue": similarity_signal["issue"],
            "matched_phrase": similarity_signal["matched_reference"],
            "reason": "Prompt is semantically similar to a known unsafe prompt.",
            "severity": 0.78,
        }

    return {
        "category": "Neutral",
        "issue": "Neutral",
        "matched_phrase": "",
        "reason": "No strong unsafe intent detected.",
        "severity": 0.10,
    }
