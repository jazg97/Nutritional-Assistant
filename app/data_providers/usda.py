from __future__ import annotations

from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx

from app.config import settings
from app.schemas import FoodProduct


class USDAFoodDataClient:
    BASE_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"

    def __init__(self) -> None:
        self.timeout = settings.request_timeout_seconds
        self.api_key = settings.usda_api_key.strip()
        self.page_size = settings.usda_page_size
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
        if not self.api_key:
            self.last_error = "USDA API key not configured."
            return []

        params = {"api_key": self.api_key}
        payload = {
            "query": query.strip(),
            "pageSize": int(page_size or self.page_size),
            "dataType": ["Branded", "Foundation", "Survey (FNDDS)"],
        }
        headers = {"User-Agent": "opencommerce-ai-assistant/0.1", "Accept": "application/json"}

        timeout = httpx.Timeout(connect=10.0, read=max(20.0, float(self.timeout)), write=10.0, pool=10.0)
        async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
            try:
                response = await client.post(self.BASE_URL, params=params, json=payload)
                self.last_status = response.status_code
                self.last_url = str(response.request.url)
                if self.debug:
                    safe_url = _redact_query_params(self.last_url, {"api_key"})
                    print(f"[DEBUG][USDA] status={self.last_status} url={safe_url} query='{query}'")
                response.raise_for_status()
                data: dict[str, Any] = response.json()
            except httpx.HTTPStatusError as exc:
                self.last_error = f"USDA HTTP {exc.response.status_code}"
                return []
            except httpx.HTTPError as exc:
                self.last_error = f"USDA network error: {exc.__class__.__name__}"
                return []

        foods = data.get("foods", []) or []
        result = [self._to_food_product(item) for item in foods if item.get("description")]
        if self.debug:
            print(f"[DEBUG][USDA] returned_products={len(result)} query='{query}'")
        if not result and not self.last_error:
            self.last_error = "No products returned by USDA FoodData Central for this query."
        return result

    @staticmethod
    def _to_food_product(item: dict[str, Any]) -> FoodProduct:
        nutrients = item.get("foodNutrients", []) or []

        kcal = _extract_nutrient(nutrients, {"Energy"})
        sugar = _extract_nutrient(nutrients, {"Sugars, total including NLEA", "Sugars, total"})
        protein = _extract_nutrient(nutrients, {"Protein"})
        fat = _extract_nutrient(nutrients, {"Total lipid (fat)"})
        sodium_mg = _extract_nutrient(nutrients, {"Sodium, Na"})
        salt_g = (sodium_mg / 1000.0) * 2.5 if sodium_mg is not None else None

        brand = str(item.get("brandOwner", "") or item.get("brandName", ""))
        desc = str(item.get("description", ""))
        fdc_id = str(item.get("fdcId", ""))
        return FoodProduct(
            code=fdc_id,
            product_name=desc,
            brands=brand,
            nutriscore_grade="",
            energy_kcal_100g=kcal,
            sugars_100g=sugar,
            proteins_100g=protein,
            fat_100g=fat,
            salt_100g=salt_g,
            ingredients_text=str(item.get("ingredients", "")),
            url=f"https://fdc.nal.usda.gov/fdc-app.html#/food-details/{fdc_id}/nutrients" if fdc_id else "",
        )


def _extract_nutrient(nutrients: list[dict[str, Any]], names: set[str]) -> float | None:
    for n in nutrients:
        name = str(n.get("nutrientName", ""))
        if name in names:
            value = n.get("value")
            try:
                if value is None:
                    return None
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def _redact_query_params(url: str, keys: set[str]) -> str:
    try:
        split = urlsplit(url)
        pairs = parse_qsl(split.query, keep_blank_values=True)
        safe_pairs = [(k, "***" if k in keys else v) for k, v in pairs]
        return urlunsplit((split.scheme, split.netloc, split.path, urlencode(safe_pairs), split.fragment))
    except Exception:
        return url
