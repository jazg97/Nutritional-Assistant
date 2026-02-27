from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


IntentType = Literal["search", "compare", "category", "other"]


class IntentPayload(BaseModel):
    intent: IntentType = "other"
    query: str = ""
    category_hint: str = ""
    max_results: int = Field(default=10, ge=1, le=50)


class Product(BaseModel):
    id: str
    title: str
    price: float
    currency_id: str
    condition: str
    permalink: str
    seller_nickname: str
    category_id: str


class FoodProduct(BaseModel):
    code: str
    product_name: str
    brands: str = ""
    nutriscore_grade: str = ""
    energy_kcal_100g: Optional[float] = None
    sugars_100g: Optional[float] = None
    proteins_100g: Optional[float] = None
    fat_100g: Optional[float] = None
    salt_100g: Optional[float] = None
    ingredients_text: str = ""
    url: str = ""
