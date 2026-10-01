"""
LLM client for the agent's "reasoning" / explanation step.

Uses a local Ollama server (http://localhost:11434) when available — no
paid API keys required. If Ollama isn't running or the model isn't pulled,
the agent falls back to a clear, template-based explanation so the
pipeline never breaks a demo.

To use Ollama:
    1. Install from https://ollama.com
    2. `ollama pull llama3.1` (or any model you prefer)
    3. Set OLLAMA_MODEL env var if you use a different model name
"""
import os
import requests

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
TIMEOUT_SECONDS = 15


def ollama_available() -> bool:
    try:
        resp = requests.get(OLLAMA_URL.replace("/api/generate", "/api/tags"), timeout=2)
        return resp.status_code == 200
    except requests.exceptions.RequestException:
        return False


def generate(prompt: str, system: str = "") -> str | None:
    """Returns generated text, or None if Ollama is unavailable/errors out."""
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "system": system,
                "stream": False,
            },
            timeout=TIMEOUT_SECONDS,
        )
        if resp.status_code == 200:
            return resp.json().get("response", "").strip()
        return None
    except requests.exceptions.RequestException:
        return None


AGENT_SYSTEM_PROMPT = (
    "You are a procurement analyst assistant. Given structured demand "
    "forecast, inventory, supplier retrieval, and risk-assessment data, "
    "write a concise (3-5 sentence) procurement recommendation explaining "
    "WHAT to order, HOW MUCH, from WHICH supplier, and WHY — referencing "
    "the specific numbers you were given. Be direct and factual, no filler."
)


def explain_recommendation(context: dict) -> str:
    """
    Turns the pipeline's structured outputs into a natural-language
    recommendation. Tries the local LLM first; falls back to a
    deterministic template if it's unavailable.
    """
    prompt = (
        f"Product: {context['product_name']} (SKU {context['sku']})\n"
        f"Forecasted demand: {context['predicted_daily_demand']} units/day "
        f"over the next {context['horizon_days']} days "
        f"(total {context['predicted_total_demand']} units), via {context['forecast_model']} model.\n"
        f"Current stock: {context['on_hand_qty']} units "
        f"({context['days_of_cover']} days of cover, stockout risk: {context['stockout_risk']}).\n"
        f"Recommended order quantity: {context['recommended_order_qty']} units.\n"
        f"Top candidate supplier: {context['supplier_name']} "
        f"(region: {context['supplier_region']}, risk: {context['risk_label']}, "
        f"reliability: {context['reliability_score']}, quality: {context['quality_score']}, "
        f"lead time: {context['lead_time_days']} days, unit price: ${context['unit_price']}).\n"
        "Write the recommendation now."
    )

    text = generate(prompt, system=AGENT_SYSTEM_PROMPT)
    if text:
        return text
    return _template_fallback(context)


def _template_fallback(c: dict) -> str:
    return (
        f"Recommend ordering {c['recommended_order_qty']} units of "
        f"{c['product_name']} ({c['sku']}). Current stock of {c['on_hand_qty']} units "
        f"provides only {c['days_of_cover']} days of cover against a forecasted demand "
        f"of {c['predicted_daily_demand']} units/day ({c['forecast_model']} model), "
        f"putting stockout risk at '{c['stockout_risk']}'. "
        f"Best-fit supplier is {c['supplier_name']} ({c['supplier_region']}), "
        f"with a '{c['risk_label']}' procurement risk rating, "
        f"{c['reliability_score']*100:.0f}% on-time delivery, "
        f"{c['quality_score']*100:.0f}% quality pass rate, a "
        f"{c['lead_time_days']}-day lead time, and a unit price of ${c['unit_price']}. "
        f"[Generated via deterministic template — connect a local Ollama model "
        f"for LLM-generated reasoning.]"
    )
