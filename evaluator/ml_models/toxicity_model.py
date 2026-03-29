import os

import requests


HF_API_TOKEN = os.environ.get("HUGGINGFACE_API_TOKEN", "").strip()
MODEL_URL = "https://api-inference.huggingface.co/models/unitary/toxic-bert"
TOXIC_LABELS = {
    "toxic",
    "severe_toxic",
    "obscene",
    "insult",
    "threat",
    "identity_hate",
}
TOXIC_KEYWORDS = {
    "hate": 0.72,
    "stupid": 0.55,
    "idiot": 0.58,
    "worthless": 0.65,
    "trash": 0.52,
    "kill": 0.80,
    "attack": 0.78,
}


def _clamp(value):
    return round(max(0.0, min(1.0, value)), 2)


def _keyword_toxicity(text):
    text_lower = (text or "").lower()
    score = 0.05
    for keyword, weight in TOXIC_KEYWORDS.items():
        if keyword in text_lower:
            score = max(score, weight)
    return _clamp(score)


def get_toxicity_probability(text, use_remote=True):
    if not (text or "").strip():
        return 0.0

    heuristic_score = _keyword_toxicity(text)
    if not use_remote or not HF_API_TOKEN:
        return heuristic_score

    try:
        response = requests.post(
            MODEL_URL,
            headers={"Authorization": f"Bearer {HF_API_TOKEN}"},
            json={"inputs": text[:512], "options": {"wait_for_model": True}},
            timeout=20,
        )
        response.raise_for_status()
        result = response.json()
        if isinstance(result, list) and result and isinstance(result[0], list):
            label_scores = {
                item.get("label", "").lower(): item.get("score", 0.0)
                for item in result[0]
                if isinstance(item, dict)
            }
            remote_score = max(
                (score for label, score in label_scores.items() if label in TOXIC_LABELS),
                default=0.0,
            )
            return _clamp(max(heuristic_score, remote_score))
    except Exception:
        return heuristic_score

    return heuristic_score
