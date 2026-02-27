from __future__ import annotations

import asyncio
from typing import Any

import httpx

from app.config import settings
from app.schemas import FoodProduct


class OpenFoodFactsClient:
    BASE_URL = "https://world.openfoodfacts.org/cgi/search.pl"

    def __init__(self) -> None:
        self.timeout = settings.request_timeout_seconds
        self.country = settings.off_country
        self.page_size = settings.off_page_size
        self.max_retries = max(1, settings.off_max_retries)
        self.debug = settings.debug_log
        self.last_error: str = ""
        self.last_status: int | None = None
        self.last_url: str = ""

    async def search_products(self, query: str, page_size: int | None = None) -> list[FoodProduct]:
        self.last_error = ""
        self.last_status = None
        self.last_url = ""
        if not query.strip():
            return []

        params = {
            "search_terms": query.strip(),
            "search_simple": "1",
            "action": "process",
            "json": "1",
            "page_size": str(page_size or self.page_size),
            "fields": ",".join(
                [
                    "code",
                    "product_name",
                    "brands",
                    "nutriscore_grade",
                    "nutriments",
                    "ingredients_text",
                    "url",
                ]
            ),
        }
        country = (self.country or "").strip().lower()
        # "world" should mean no country filter.
        if country and country not in {"world", "all", "*"}:
            params["tagtype_0"] = "countries"
            params["tag_contains_0"] = "contains"
            params["tag_0"] = country
        headers = {"User-Agent": "opencommerce-ai-assistant/0.1", "Accept": "application/json"}

        timeout = httpx.Timeout(connect=10.0, read=max(20.0, float(self.timeout)), write=10.0, pool=10.0)
        async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
            payload: dict[str, Any] | None = None
            for attempt in range(1, self.max_retries + 1):
                try:
                    response = await client.get(self.BASE_URL, params=params)
                    self.last_status = response.status_code
                    self.last_url = str(response.request.url)
                    if self.debug:
                        print(
                            f"[DEBUG][OFF] attempt={attempt}/{self.max_retries} "
                            f"status={self.last_status} url={self.last_url}"
                        )
                    response.raise_for_status()
                    payload = response.json()
                    self.last_error = ""
                    break
                except httpx.HTTPStatusError as exc:
                    self.last_error = f"OpenFoodFacts HTTP {exc.response.status_code}"
                    return []
                except httpx.ReadTimeout:
                    self.last_error = "OpenFoodFacts network error: ReadTimeout"
                    if self.debug:
                        print(f"[DEBUG][OFF] attempt={attempt}/{self.max_retries} timeout for query='{query}'")
                    if attempt < self.max_retries:
                        await asyncio.sleep(0.6 * attempt)
                        continue
                    return []
                except httpx.HTTPError as exc:
                    self.last_error = f"OpenFoodFacts network error: {exc.__class__.__name__}"
                    if self.debug:
                        print(f"[DEBUG][OFF] http_error={exc.__class__.__name__} query='{query}'")
                    return []
            if payload is None:
                return []

        products = payload.get("products", [])
        result = [self._to_food_product(item) for item in products if item.get("product_name")]
        if self.debug:
            print(f"[DEBUG][OFF] returned_products={len(result)} query='{query}'")
            for idx, p in enumerate(result[:3], start=1):
                print(
                    f"[DEBUG][OFF] sample#{idx} name='{p.product_name}' "
                    f"nutriscore='{p.nutriscore_grade}' sugar_100g={p.sugars_100g} protein_100g={p.proteins_100g}"
                )
        if not result and not self.last_error:
            self.last_error = "No products returned by OpenFoodFacts for this query."
        return result

    @staticmethod
    def _to_food_product(item: dict[str, Any]) -> FoodProduct:
        nutriments = item.get("nutriments", {}) or {}
        return FoodProduct(
            code=str(item.get("code", "")),
            product_name=str(item.get("product_name", "")),
            brands=str(item.get("brands", "")),
            nutriscore_grade=str(item.get("nutriscore_grade", "")).upper(),
            energy_kcal_100g=_to_float(nutriments.get("energy-kcal_100g")),
            sugars_100g=_to_float(nutriments.get("sugars_100g")),
            proteins_100g=_to_float(nutriments.get("proteins_100g")),
            fat_100g=_to_float(nutriments.get("fat_100g")),
            salt_100g=_to_float(nutriments.get("salt_100g")),
            ingredients_text=str(item.get("ingredients_text", "")),
            url=str(item.get("url", "")),
        )


def _to_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
