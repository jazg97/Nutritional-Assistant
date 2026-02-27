import json

from app.schemas import IntentPayload


def parse_intent_output(raw_text: str, fallback_query: str) -> IntentPayload:
    text = (raw_text or "").strip()
    if not text:
        return heuristic_intent(fallback_query)

    try:
        payload = json.loads(text)
        return IntentPayload.model_validate(payload)
    except Exception:
        return heuristic_intent(fallback_query)


def heuristic_intent(user_text: str) -> IntentPayload:
    content = (user_text or "").strip()
    lowered = content.lower()

    if lowered in {"hi", "hello", "hey", "hola", "buenas", "ola"}:
        return IntentPayload(intent="other", query="", max_results=5)

    if any(word in lowered for word in ["compar", "vs", "versus"]):
        return IntentPayload(intent="compare", query=content, max_results=8)
    if any(word in lowered for word in ["categoria", "category", "tipo"]):
        return IntentPayload(intent="category", query=content, max_results=12)
    if any(word in lowered for word in ["precio", "comprar", "zapat", "camisa", "laptop", "telefono"]):
        return IntentPayload(intent="search", query=content, max_results=12)

    return IntentPayload(intent="search", query=content, max_results=10)
