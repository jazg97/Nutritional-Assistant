from __future__ import annotations

from typing import Any

import httpx

from app.config import settings
from app.schemas import Product


class MercadoLibreClient:
    BASE_URL = "https://api.mercadolibre.com"

    def __init__(self) -> None:
        self.timeout = settings.request_timeout_seconds
        self.site_id = settings.meli_site_id
        self.fallback_sites = [
            s.strip().upper()
            for s in settings.meli_fallback_sites.split(",")
            if s.strip()
        ]
        self.access_token = settings.meli_access_token.strip()
        self.default_limit = settings.meli_items_limit
        self.last_error: str = ""

    async def search_products(self, query: str, limit: int | None = None) -> list[Product]:
        self.last_error = ""
        if not query.strip():
            return []

        params = {"q": query.strip(), "limit": str(limit or self.default_limit)}
        sites = [self.site_id] + [s for s in self.fallback_sites if s != self.site_id]
        headers = {"User-Agent": "opencommerce-ai-assistant/0.1", "Accept": "application/json"}
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"

        async with httpx.AsyncClient(timeout=self.timeout, headers=headers) as client:
            for site in sites:
                url = f"{self.BASE_URL}/sites/{site}/search"
                try:
                    response = await client.get(url, params=params)
                    response.raise_for_status()
                    data: dict[str, Any] = response.json()
                    items = [self._to_product(item) for item in data.get("results", [])]
                    if items:
                        return items
                    self.last_error = f"No results for site={site}"
                except httpx.HTTPStatusError as exc:
                    body = (exc.response.text or "")[:120].replace("\n", " ")
                    self.last_error = f"MercadoLibre site={site} HTTP {exc.response.status_code}: {body}"
                except httpx.HTTPError as exc:
                    self.last_error = f"MercadoLibre network error: {exc.__class__.__name__}"

        return []

    @staticmethod
    def _to_product(item: dict[str, Any]) -> Product:
        seller = item.get("seller", {})
        return Product(
            id=str(item.get("id", "")),
            title=str(item.get("title", "")),
            price=float(item.get("price", 0.0) or 0.0),
            currency_id=str(item.get("currency_id", "")),
            condition=str(item.get("condition", "")),
            permalink=str(item.get("permalink", "")),
            seller_nickname=str(seller.get("nickname", "unknown")),
            category_id=str(item.get("category_id", "")),
        )
