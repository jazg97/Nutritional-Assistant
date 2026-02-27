from pathlib import Path
import os

import httpx
from dotenv import load_dotenv


def print_resp(tag: str, resp: httpx.Response) -> None:
    body = (resp.text or "").replace("\n", " ")[:220]
    print(f"{tag}: {resp.status_code} | {body}")


def main() -> None:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    token = os.getenv("MELI_ACCESS_TOKEN", "").strip()
    site = os.getenv("MELI_SITE_ID", "MLB").strip() or "MLB"

    headers = {"Accept": "application/json", "User-Agent": "opencommerce-ai-assistant/0.1"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    print(f"has_token={bool(token)} site={site}")

    try:
        with httpx.Client(timeout=20, headers=headers) as client:
            me = client.get("https://api.mercadolibre.com/users/me")
            print_resp("users/me", me)

            if me.status_code == 200:
                user_id = str(me.json().get("id", ""))
                items = client.get(
                    f"https://api.mercadolibre.com/users/{user_id}/items/search",
                    params={"limit": "5"},
                )
                print_resp("users/{id}/items/search", items)

            public_search = client.get(
                f"https://api.mercadolibre.com/sites/{site}/search",
                params={"q": "notebook", "limit": "5"},
            )
            print_resp("sites/{site}/search", public_search)
    except Exception as exc:
        print(f"network_error={exc.__class__.__name__}: {str(exc)[:200]}")


if __name__ == "__main__":
    main()
