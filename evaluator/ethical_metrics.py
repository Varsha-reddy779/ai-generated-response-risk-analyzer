def _clamp(value, minimum=0.0, maximum=1.0):
    return max(minimum, min(maximum, value))


def _normalize_percentage(value):
    return round(_clamp((value or 0) / 100.0), 2)


_LATEST_METRICS = {
    "bias": 0.0,
    "safety": 0.0,
    "fairness": 0.0,
    "toxicity": 0.0,
    "privacy": 0.0,
    "hallucination": 0.0,
    "ethical_score": 0.0,
}


def build_ethical_metrics(payload):
    scores = payload.get("scores", {})

    bias = _normalize_percentage(scores.get("bias"))
    safety = _normalize_percentage(scores.get("safety"))
    toxicity = _normalize_percentage(scores.get("toxicity"))
    privacy = _normalize_percentage(scores.get("privacy"))
    fairness = _normalize_percentage(scores.get("fairness"))
    hallucination = _normalize_percentage(scores.get("hallucination"))

    return {
        "bias": bias,
        "safety": safety,
        "fairness": fairness,
        "toxicity": toxicity,
        "privacy": privacy,
        "hallucination": hallucination,
        "ethical_score": _normalize_percentage(payload.get("ethical_score")),
        "confidence_score": round(payload.get("confidence_score", 0.0), 2),
    }


def store_latest_metrics(metrics):
    _LATEST_METRICS.update(metrics)


def get_latest_metrics():
    return dict(_LATEST_METRICS)
