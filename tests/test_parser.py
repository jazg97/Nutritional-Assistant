from app.llm.parser import heuristic_intent, parse_intent_output


def test_parse_intent_output_valid_json():
    raw = '{"intent":"search","query":"running shoes","category_hint":"","max_results":7}'
    parsed = parse_intent_output(raw, fallback_query="x")
    assert parsed.intent == "search"
    assert parsed.query == "running shoes"
    assert parsed.max_results == 7


def test_parse_intent_output_invalid_json_fallback():
    parsed = parse_intent_output("not-json", fallback_query="compare iphone vs samsung")
    assert parsed.intent == "compare"


def test_heuristic_intent_default_search():
    parsed = heuristic_intent("quiero comprar zapatillas")
    assert parsed.intent == "search"
