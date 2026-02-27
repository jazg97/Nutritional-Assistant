import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.data_providers.openfoodfacts import OpenFoodFactsClient


async def main() -> None:
    client = OpenFoodFactsClient()
    for term in ["coca cola", "chocolate", "yogurt"]:
        products = await client.search_products(term, page_size=5)
        print(f"query={term} products={len(products)}")
        print(f"last_error={client.last_error}")
        for item in products[:2]:
            print(
                f"- {item.product_name} | nutriscore={item.nutriscore_grade} "
                f"| sugar_100g={item.sugars_100g} | protein_100g={item.proteins_100g}"
            )
        print("---")


if __name__ == "__main__":
    asyncio.run(main())
