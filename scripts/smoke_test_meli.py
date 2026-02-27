import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.data_providers.mercadolibre import MercadoLibreClient


async def main() -> None:
    client = MercadoLibreClient()
    products = await client.search_products("zapatillas running", limit=5)
    print(f"products={len(products)}")
    print(f"last_error={client.last_error}")
    if products:
        for p in products[:3]:
            print(f"- {p.title} | {p.price} {p.currency_id}")


if __name__ == "__main__":
    asyncio.run(main())
