from __future__ import annotations

import os
from urllib.parse import urlparse

from openai import OpenAI

from app.config import settings
from app.llm.parser import parse_intent_output
from app.llm.prompts import INTENT_PROMPT
from app.schemas import IntentPayload


class IntentExtractor:
    def __init__(self) -> None:
        self.model = settings.openai_model
        self.client = None
        self.last_source = "fallback"
        if settings.openai_api_key:
            kwargs = {"api_key": settings.openai_api_key}
            if self._is_valid_http_url(settings.openai_base_url):
                kwargs["base_url"] = settings.openai_base_url
            else:
                # Let OpenAI SDK use its default URL when direct API is intended.
                os.environ.pop("OPENAI_BASE_URL", None)
            self.client = OpenAI(**kwargs)

    def extract(self, user_text: str) -> IntentPayload:
        if not self.client:
            self.last_source = "fallback"
            return parse_intent_output("", fallback_query=user_text)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.0,
                max_tokens=240,
                messages=[
                    {"role": "system", "content": INTENT_PROMPT},
                    {"role": "user", "content": user_text},
                ],
            )
            output = response.choices[0].message.content or ""
            self.last_source = "llm"
            return parse_intent_output(output, fallback_query=user_text)
        except Exception:
            # Keep the app available even if provider config/network fails.
            self.last_source = "fallback"
            return parse_intent_output("", fallback_query=user_text)

    @staticmethod
    def _is_valid_http_url(value: str) -> bool:
        if not value:
            return False
        parsed = urlparse(value)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
