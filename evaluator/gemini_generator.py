import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()
GEMINI_API_KEY = (
    os.environ.get("GEMINI_API_KEY", "").strip()
    or os.environ.get("GOOGLE_API_KEY", "").strip()
)
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash").strip()
REQUEST_TIMEOUT_SECONDS = 30
RETRY_DELAY_SECONDS = 1.5

DEFAULT_MODELS = [
    os.environ.get("OPENROUTER_MODEL", "").strip(),
    "google/gemini-2.0-flash-exp:free",
    "google/gemini-flash-1.5",
    "meta-llama/llama-3.1-8b-instruct",
]
MODEL_CANDIDATES = [model for model in DEFAULT_MODELS if model]


def _headers():
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "Ethical AI Evaluator",
    }


def _messages(prompt):
    return [
        {
            "role": "system",
            "content": "You are a responsible AI. Refuse harmful or illegal requests.",
        },
        {"role": "user", "content": prompt},
    ]


def _extract_error_message(result):
    error = result.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if message:
            return message
    if isinstance(error, str):
        return error
    return "Unknown API error."


def _try_gemini_direct(prompt):
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "systemInstruction": {
            "parts": [{"text": "You are a responsible AI. Refuse harmful or illegal requests."}]
        },
    }

    try:
        response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.Timeout:
        return None, f"Request timed out for Gemini model '{GEMINI_MODEL}'."
    except requests.RequestException as exc:
        return None, f"Network error for Gemini model '{GEMINI_MODEL}': {exc}"

    try:
        result = response.json()
    except ValueError:
        return None, f"Invalid Gemini API response (status {response.status_code})."

    candidates = result.get("candidates") or []
    if response.ok and candidates:
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(part.get("text", "") for part in parts).strip()
        if text:
            return text, None
        return None, f"Empty content returned by Gemini model '{GEMINI_MODEL}'."

    return None, _extract_error_message(result)


def _try_model(prompt, model):
    payload = {"model": model, "messages": _messages(prompt)}

    try:
        response = requests.post(
            OPENROUTER_URL,
            headers=_headers(),
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.Timeout:
        return None, f"Request timed out for model '{model}'."
    except requests.RequestException as exc:
        return None, f"Network error for model '{model}': {exc}"

    try:
        result = response.json()
    except ValueError:
        return None, f"Invalid API response for model '{model}' (status {response.status_code})."

    choices = result.get("choices") or []
    if response.ok and choices:
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if content:
            return content, None
        return None, f"Empty content returned by model '{model}'."

    return None, _extract_error_message(result)


def generate_response(prompt):
    clean_prompt = (prompt or "").strip()
    if not clean_prompt:
        return "Prompt is empty. Please enter a question."

    if GEMINI_API_KEY:
        content, error_message = _try_gemini_direct(clean_prompt)
        if content:
            time.sleep(1)
            return content

        if "user not found" not in (error_message or "").lower():
            return f"Model request failed. {error_message}"

    if not API_KEY:
        return "API key not configured. Set GEMINI_API_KEY or OPENROUTER_API_KEY in environment."

    last_error = "No available model succeeded."

    for index, model in enumerate(MODEL_CANDIDATES):
        content, error_message = _try_model(clean_prompt, model)
        if content:
            time.sleep(1)
            return content

        last_error = error_message or last_error

        if index < len(MODEL_CANDIDATES) - 1:
            time.sleep(RETRY_DELAY_SECONDS)

    if "user not found" in last_error.lower():
        return "Model request failed. OpenRouter API key is invalid or expired. Set a valid OPENROUTER_API_KEY or GEMINI_API_KEY."

    return f"Model request failed. {last_error}"
