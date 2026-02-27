from __future__ import annotations

import json
import os
import re
from urllib.parse import urlparse

from openai import OpenAI

from app.config import settings
from app.llm.prompts import (
    CATALOG_GROUNDED_SYSTEM_PROMPT,
    FOOD_QUERY_EXTRACTION_SYSTEM_PROMPT,
    GENERAL_NUTRITION_SYSTEM_PROMPT,
)


class ChatResponder:
    NATURAL_FOODS = {
        "apple",
        "orange",
        "banana",
        "mango",
        "grape",
        "pear",
        "pineapple",
        "watermelon",
        "strawberry",
        "blueberry",
        "avocado",
        "broccoli",
        "spinach",
        "carrot",
        "tomato",
        "potato",
        "onion",
        "garlic",
        "rice",
        "egg",
        "chicken",
        "beef",
        "fish",
    }
    def __init__(self) -> None:
        self.model = settings.openai_model
        self.client = None
        self.last_source = "fallback"
        self.last_error = ""

        if settings.openai_api_key:
            kwargs = {"api_key": settings.openai_api_key}
            if self._is_valid_http_url(settings.openai_base_url):
                kwargs["base_url"] = settings.openai_base_url
            else:
                os.environ.pop("OPENAI_BASE_URL", None)
            self.client = OpenAI(**kwargs)

    def reply(self, user_text: str, history: list[dict] | None = None) -> str:
        return self._reply_with_messages(
            messages=self._with_history(
                base_messages=[{"role": "system", "content": GENERAL_NUTRITION_SYSTEM_PROMPT}],
                history=history,
                user_text=user_text,
            )
        )

    def reply_with_context(self, user_text: str, context: str, history: list[dict] | None = None) -> str:
        return self._reply_with_messages(
            messages=self._with_history(
                base_messages=[{"role": "system", "content": CATALOG_GROUNDED_SYSTEM_PROMPT}],
                history=history,
                user_text=f"User request: {user_text}\n\nCATALOG_CONTEXT:\n{context}",
            )
        )

    def _reply_with_messages(self, messages: list[dict]) -> str:
        self.last_error = ""
        if not self.client:
            self.last_source = "fallback"
            return "OPENAI_API_KEY is not configured."

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                max_tokens=500,
                messages=messages,
            )
            self.last_source = "llm"
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:
            self.last_source = "fallback"
            self.last_error = f"{exc.__class__.__name__}"
            return "OpenAI request failed. Check network/API key and try again."

    def extract_food_query(self, user_text: str, history: list[dict] | None = None, use_history: bool = False) -> dict:
        fallback = self._fallback_extract_food_query(user_text)
        if not self.client:
            return fallback

        try:
            if use_history:
                messages = self._with_history(
                    base_messages=[{"role": "system", "content": FOOD_QUERY_EXTRACTION_SYSTEM_PROMPT}],
                    history=history,
                    user_text=user_text,
                )
            else:
                messages = [
                    {"role": "system", "content": FOOD_QUERY_EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_text},
                ]
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.0,
                max_tokens=120,
                messages=messages,
            )
            raw = (response.choices[0].message.content or "").strip()
            parsed = json.loads(raw)
            mode = str(parsed.get("mode", "")).strip().lower()
            food_query = str(parsed.get("food_query", "")).strip().lower()
            compare_items = parsed.get("compare_items", []) or []
            if mode not in {"catalog", "general", "memory", "compare", "correction"}:
                return fallback
            if not food_query:
                return fallback
            cleaned_items: list[str] = []
            for item in compare_items:
                s = str(item).strip().lower()
                if s and s not in cleaned_items:
                    cleaned_items.append(s)
            out = {"mode": mode, "food_query": food_query, "compare_items": cleaned_items[:4]}
            if self.is_natural_food_request(out):
                out["mode"] = "general"
            return out
        except Exception:
            return fallback

    @staticmethod
    def _with_history(base_messages: list[dict], history: list[dict] | None, user_text: str) -> list[dict]:
        if not history:
            return base_messages + [{"role": "user", "content": user_text}]

        recent = history[-6:]
        normalized: list[dict] = []
        for msg in recent:
            role = str(msg.get("role", "")).strip().lower()
            if role not in {"user", "assistant"}:
                continue
            content = str(msg.get("content", "")).strip()
            if content:
                normalized.append({"role": role, "content": content[:1500]})
        normalized.append({"role": "user", "content": user_text})
        return base_messages + normalized

    @staticmethod
    def _is_valid_http_url(value: str) -> bool:
        if not value:
            return False
        parsed = urlparse(value)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

    @staticmethod
    def _fallback_extract_food_query(text: str) -> dict:
        lower = (text or "").lower()
        if any(x in lower for x in [" vs ", " versus ", " or ", "compare ", "which has better", "better than"]):
            items = ChatResponder._extract_compare_items(lower)
            if len(items) >= 2:
                return {"mode": "compare", "food_query": " ".join(items), "compare_items": items}

        if any(
            x in lower
            for x in [
                "didn t ask about that",
                "didn't ask about that",
                "not related to my query",
                "not what i asked",
                "you are wrong",
                "your answer wasn't related",
                "answer was not related",
                "off topic",
            ]
        ):
            return {"mode": "correction", "food_query": "correction request", "compare_items": []}

        if any(
            x in lower
            for x in [
                "products i asked",
                "what products",
                "list the products",
                "conversation history",
                "earlier products",
            ]
        ):
            return {"mode": "memory", "food_query": "conversation products", "compare_items": []}

        cleaned = re.sub(r"[^a-z0-9\s]", " ", lower)
        cleaned = re.sub(
            r"\b(i|want|to|know|if|can you|could you|please|tell me|about|how many|what are|what is|there|in|the|a|an|bottle|facts|nutrition|nutritional|for|of)\b",
            " ",
            cleaned,
        )
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        tokens = cleaned.split()
        query = " ".join(tokens[:5]) if tokens else lower.strip() or "food"

        if any(x in lower for x in ["brand", "bar", "bottle", "snickers", "doritos", "gatorade", "coca", "pepsi"]):
            mode = "catalog"
        elif any(
            x in lower
            for x in ["more calories than", "vs", "versus", "is it good", "healthy", "calories in an", "calories in a"]
        ):
            mode = "general"
        else:
            mode = "catalog"

        out = {"mode": mode, "food_query": query, "compare_items": []}
        if ChatResponder.is_natural_food_request(out):
            out["mode"] = "general"
        return out

    @staticmethod
    def _extract_compare_items(lower_text: str) -> list[str]:
        text = re.sub(r"[^a-z0-9\s]", " ", lower_text)
        text = re.sub(r"\b(which has better|compare|versus|vs|better than)\b", " ", text)
        parts = re.split(r"\bor\b|\band\b", text)
        items: list[str] = []
        for part in parts:
            cleaned = re.sub(
                r"\b(i|want|to|know|if|the|a|an|of|for|nutrition|facts|calories|electrolytes|has|more|better|which)\b",
                " ",
                part,
            )
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            if cleaned and cleaned not in items:
                items.append(cleaned[:40])
        return items[:4]

    @staticmethod
    def is_natural_food_request(extracted: dict) -> bool:
        mode = str(extracted.get("mode", "")).lower()
        if mode in {"memory", "correction"}:
            return False
        hay = " ".join(
            [str(extracted.get("food_query", ""))]
            + [str(x) for x in (extracted.get("compare_items", []) or [])]
        ).lower()
        tokens = set(re.findall(r"[a-z]+", hay))
        return len(tokens & ChatResponder.NATURAL_FOODS) > 0
