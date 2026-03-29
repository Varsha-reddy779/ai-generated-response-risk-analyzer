import math
import os
from difflib import SequenceMatcher

import requests


HF_API_TOKEN = os.environ.get("HUGGINGFACE_API_TOKEN", "").strip()
MODEL_URL = (
    "https://api-inference.huggingface.co/pipeline/feature-extraction/"
    "sentence-transformers/all-MiniLM-L6-v2"
)
SIMILARITY_THRESHOLD = 0.75

UNSAFE_REFERENCES = {
    "Violence": [
        "how to build a bomb",
        "how to make a weapon",
        "how to hurt another person",
    ],
    "Illegal Activity": [
        "how to make illegal drugs",
        "how to create fake documents",
        "how to commit fraud",
    ],
    "Self Harm": [
        "how to end their life",
        "how to commit suicide",
        "self harm instructions",
    ],
    "Bias / Hate": [
        "why are some races superior",
        "why are women bad at engineering",
        "why are immigrants bad",
    ],
}

_EMBEDDING_CACHE = {}


def _clamp(value):
    return round(max(0.0, min(1.0, value)), 2)


def _cosine_similarity(vector_a, vector_b):
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    magnitude_a = math.sqrt(sum(a * a for a in vector_a))
    magnitude_b = math.sqrt(sum(b * b for b in vector_b))
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    return dot_product / (magnitude_a * magnitude_b)


def _normalize_embedding(raw_embedding):
    if not raw_embedding:
        return []
    if isinstance(raw_embedding[0], list):
        token_count = len(raw_embedding)
        dimensions = len(raw_embedding[0])
        return [
            sum(token_vector[index] for token_vector in raw_embedding) / token_count
            for index in range(dimensions)
        ]
    return raw_embedding


def _fetch_embedding(text):
    if text in _EMBEDDING_CACHE:
        return _EMBEDDING_CACHE[text]

    response = requests.post(
        MODEL_URL,
        headers={"Authorization": f"Bearer {HF_API_TOKEN}"},
        json={"inputs": text[:256], "options": {"wait_for_model": True}},
        timeout=20,
    )
    response.raise_for_status()
    embedding = _normalize_embedding(response.json())
    _EMBEDDING_CACHE[text] = embedding
    return embedding


def _fallback_similarity(prompt, reference):
    prompt_lower = prompt.lower()
    reference_lower = reference.lower()
    sequence_score = SequenceMatcher(None, prompt_lower, reference_lower).ratio()
    prompt_tokens = set(prompt_lower.split())
    reference_tokens = set(reference_lower.split())
    overlap_score = len(prompt_tokens & reference_tokens) / max(len(reference_tokens), 1)
    return max(sequence_score, overlap_score)


def get_similarity_signal(prompt, use_remote=True):
    prompt = (prompt or "").strip()
    if not prompt:
        return {
            "max_similarity": 0.0,
            "matched_reference": "",
            "issue": "Neutral",
        }

    best_match = {
        "max_similarity": 0.0,
        "matched_reference": "",
        "issue": "Neutral",
    }

    use_embeddings = use_remote and bool(HF_API_TOKEN)
    prompt_embedding = None

    try:
        if use_embeddings:
            prompt_embedding = _fetch_embedding(prompt)
    except Exception:
        use_embeddings = False

    for issue, references in UNSAFE_REFERENCES.items():
        for reference in references:
            if use_embeddings and prompt_embedding:
                try:
                    reference_embedding = _fetch_embedding(reference)
                    similarity = _cosine_similarity(prompt_embedding, reference_embedding)
                except Exception:
                    similarity = _fallback_similarity(prompt, reference)
            else:
                similarity = _fallback_similarity(prompt, reference)

            if similarity > best_match["max_similarity"]:
                best_match = {
                    "max_similarity": _clamp(similarity),
                    "matched_reference": reference,
                    "issue": issue,
                }

    return best_match
