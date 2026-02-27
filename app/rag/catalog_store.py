from __future__ import annotations

import re
from collections import Counter

from app.schemas import Product


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class CatalogStore:
    """
    Lightweight lexical retriever for Day 1.
    Replaces embeddings while keeping the same retrieval interface.
    """

    def rerank(self, query: str, products: list[Product], top_k: int = 5) -> list[Product]:
        if not products:
            return []

        q_tokens = Counter(_tokenize(query))
        scored: list[tuple[int, Product]] = []

        for product in products:
            p_tokens = Counter(_tokenize(f"{product.title} {product.category_id}"))
            overlap = sum((q_tokens & p_tokens).values())
            scored.append((overlap, product))

        ranked = sorted(scored, key=lambda x: x[0], reverse=True)
        return [item[1] for item in ranked[:top_k]]
