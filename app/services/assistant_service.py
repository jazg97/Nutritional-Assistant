from __future__ import annotations

import asyncio
import re

from app.config import settings
from app.data_providers.usda import USDAFoodDataClient
from app.schemas import FoodProduct
from app.llm.responder import ChatResponder


class AssistantService:
    def __init__(self) -> None:
        self.chat = ChatResponder()
        self.usda = USDAFoodDataClient()
        self.debug = settings.debug_log

    async def answer(
        self,
        user_text: str,
        history: list[dict] | None = None,
        allow_correction_retry: bool = True,
    ) -> str:
        text = (user_text or "").strip()
        if not text:
            return "Send a message to chat with the model."
        if self.debug:
            print(f"[DEBUG][SERVICE] user_text='{text}'")

        lowered = text.lower()
        if lowered in {"hi", "hello", "hey", "hola", "buenas", "ola"}:
            return (
                "[source: ux]\n\n"
                "What would you like to do?\n\n"
                "1. Compare products\n"
                "2. Check nutrition facts for one product\n"
                "3. Tell me your goal\n\n"
                "You can include a goal like: lower calories, lower sugar, higher protein, or lower sodium."
            )

        extraction = self.chat.extract_food_query(text, history=history, use_history=False)
        if self.chat.is_natural_food_request(extraction):
            extraction["mode"] = "general"
        elif extraction.get("mode") in {"general", "catalog"} and self._should_force_catalog_mode(text):
            extraction["mode"] = "catalog"
        mode = extraction.get("mode", "catalog")
        search_query = extraction.get("food_query", text)
        compare_items = extraction.get("compare_items", []) or []
        session_state = self._build_session_state(history)
        goal = self._infer_goal(text, session_state)
        if self.debug:
            print(
                f"[DEBUG][SERVICE] extracted_mode='{mode}' extracted_query='{search_query}' "
                f"compare_items={compare_items} goal='{goal}'"
            )

        if mode == "memory":
            if not session_state["products"]:
                return (
                    "[source: memory]\n\n"
                    "I do not have earlier product queries in this chat yet. "
                    "If you want, I can start by comparing two products or analyzing one product label."
                )
            lines = ["[source: memory]", "", "Here is what we have covered so far:"]
            lines.append("- Products discussed: " + ", ".join(session_state["products"]))
            lines.append("- Last active goal: " + session_state["goal"])
            lines.append("")
            lines.append("If you want, I can continue with that same goal or switch to a new one.")
            return "\n".join(lines)

        if mode == "correction" and allow_correction_retry:
            previous_query = self._latest_actionable_user_query(history)
            if not previous_query:
                return "[source: correction]\n\nUnderstood. Please restate what product(s) you want me to analyze."
            if self.debug:
                print(f"[DEBUG][SERVICE] correction_target='{previous_query}'")
            return await self.answer(
                previous_query,
                history=history,
                allow_correction_retry=False,
            )

        if mode == "general":
            if self._needs_goal_clarification(text):
                return (
                    "[source: clarification]\n\n"
                    "What do you mean by better here: lower calories, lower sugar, higher protein, or lower sodium? "
                    "If you want, I can default to lower calories."
                )
            answer = self.chat.reply(
                f"SESSION_STATE: {self._session_state_text(session_state)}\n\n"
                f"User question: {text}\n"
                "Answer as a nutrition assistant with concise, practical advice. "
                "If user asks numbers, clarify they are approximate unless label data is provided.",
                history=history,
            )
            return f"[source: {self.chat.last_source}]\n\n{answer}"

        if mode == "compare":
            if self._needs_goal_clarification(text):
                return (
                    "[source: clarification]\n\n"
                    "Before I compare them, what should 'better' mean here: lower calories, lower sugar, higher protein, or lower sodium?"
                )
            if len(compare_items) < 2:
                compare_items = self._split_compare_items(search_query)
            if len(compare_items) < 2:
                answer = self.chat.reply(
                    "Ask one short clarification question to identify the 2 products to compare.",
                    history=history,
                )
                return f"[source: {self.chat.last_source}]\n\n{answer}"

            grouped_context = []
            total_hits = 0
            tasks = [self._search_item_for_compare(item_query) for item_query in compare_items[:4]]
            compare_results = await asyncio.gather(*tasks)
            best_rows: list[tuple[str, FoodProduct]] = []
            explanations = []
            for item_query, found, provider, provider_error, provider_status in compare_results:
                filtered, match_meta = self._filter_relevant_products(item_query, found)
                total_hits += len(filtered)
                if self.debug:
                    print(
                        f"[DEBUG][SERVICE] compare_item='{item_query}' raw_hits={len(found)} filtered_hits={len(filtered)} "
                        f"provider='{provider}' error='{provider_error}' status={provider_status}"
                    )
                if not filtered:
                    grouped_context.append(f"ITEM_QUERY: {item_query}\n- No relevant matches in catalog.")
                    continue
                explanations.append(
                    f"- {item_query}: {match_meta['confidence']} confidence, {match_meta['explanation']}, source={provider}"
                )
                lines = []
                for idx, item in enumerate(filtered[:3], start=1):
                    if idx == 1:
                        best_rows.append((item_query, item))
                    lines.append(
                        f"{idx}. {item.product_name} | brand={item.brands or 'n/a'} | "
                        f"kcal_100g={item.energy_kcal_100g} | sugar_100g={item.sugars_100g} | "
                        f"protein_100g={item.proteins_100g} | fat_100g={item.fat_100g} | salt_100g={item.salt_100g} | url={item.url}"
                    )
                grouped_context.append(f"ITEM_QUERY: {item_query} | SOURCE: {provider}\n" + "\n".join(lines))

            if total_hits == 0:
                answer = self.chat.reply(
                    f"User question: {text}\nNo catalog matches found. Give general comparison guidance and ask user to provide exact product names.",
                    history=history,
                )
                return f"[source: {self.chat.last_source}]\n\n{answer}"

            table = self._format_comparison_table(best_rows, goal)
            match_block = "Match quality\n\n" + "\n".join(explanations)
            compare_context = "\n\n".join(grouped_context)
            answer = self.chat.reply_with_context(
                user_text=(
                    f"SESSION_STATE: {self._session_state_text(session_state)}\n"
                    f"{text}\n"
                    "Compare the requested items side-by-side using catalog values when available. "
                    f"The comparison goal is '{goal}'. "
                    "If data is missing for an item, explicitly say so."
                ),
                context=compare_context,
                history=history,
            )
            answer = self._ensure_natural_answer(answer, best_rows, goal, is_compare=True)
            compare_source = f"{self.chat.last_source} + usda-compare"
            if self.chat.last_source != "llm" and self.chat.last_error:
                compare_source += f" ({self.chat.last_error})"
            return f"[source: {compare_source}]\n\n{answer}\n\n{table}\n\n{match_block}"

        products, source, source_error, source_status = await self._search_usda(search_query, page_size=12)
        products, match_meta = self._filter_relevant_products(search_query, products)
        if self.debug:
            print(
                f"[DEBUG][SERVICE] products={len(products)} source='{source}' "
                f"error='{source_error}' status={source_status}"
            )
        if not products:
            # Retry once with a shorter query to improve catalog hit-rate.
            short_query = " ".join(search_query.split()[:3]).strip()
            if short_query and short_query != search_query:
                products, source, source_error, source_status = await self._search_usda(short_query, page_size=12)
                products, match_meta = self._filter_relevant_products(short_query, products)
                if self.debug:
                    print(
                        f"[DEBUG][SERVICE] retry_query='{short_query}' products={len(products)} "
                        f"source='{source}' error='{source_error}' status={source_status}"
                    )
            if products:
                search_query = short_query
            else:
                # If catalog misses, still provide useful nutrition guidance via LLM.
                answer = self.chat.reply(
                    f"User question: {text}\n"
                    "Answer as a nutrition assistant. "
                    "If exact product facts are unknown, state that briefly and provide general guidance.",
                    history=history,
                )
                if self.chat.last_source == "llm":
                    if self.debug:
                        print("[DEBUG][SERVICE] response_source='llm' (no catalog hits)")
                    return f"[source: llm]\n\n{answer}"
                details = source_error or self.usda.last_error or "No matching products in catalog."
                if self.debug:
                    print("[DEBUG][SERVICE] response_source='usda' (llm unavailable)")
                return f"[source: usda]\n\nNo catalog results. Details: {details}"

        if self._needs_disambiguation(search_query, products):
            options = [f"- {p.product_name}" for p in products[:4]]
            return (
                "[source: disambiguation]\n\n"
                "I found multiple plausible product variants. Which one do you mean?\n\n"
                + "\n".join(options)
            )

        context_lines = []
        single_best = []
        for idx, item in enumerate(products[:6], start=1):
            if idx == 1:
                single_best.append((search_query, item))
            context_lines.append(
                f"{idx}. {item.product_name} | brand={item.brands or 'n/a'} | "
                f"kcal_100g={item.energy_kcal_100g} | sugar_100g={item.sugars_100g} | "
                f"protein_100g={item.proteins_100g} | fat_100g={item.fat_100g} | salt_100g={item.salt_100g} | url={item.url}"
            )
        context = "\n".join(context_lines)
        table = self._format_comparison_table(single_best, goal)

        answer = self.chat.reply_with_context(
            f"SESSION_STATE: {self._session_state_text(session_state)}\n"
            f"User request: {text}\n"
            f"The main goal is '{goal}'.",
            context=context,
            history=history,
        )
        if self.chat.last_source == "llm":
            if self.debug:
                print("[DEBUG][SERVICE] response_source='llm + usda'")
            match_block = (
                "Match quality\n\n"
                f"- confidence: {match_meta['confidence']}\n"
                f"- explanation: {match_meta['explanation']}\n"
                f"- source: {source}"
            )
            answer = self._ensure_natural_answer(answer, single_best, goal, is_compare=False)
            return f"[source: llm + usda]\n\n{answer}\n\n{table}\n\n{match_block}"

        details = f" ({self.chat.last_error})" if self.chat.last_error else ""
        if self.debug:
            print("[DEBUG][SERVICE] response_source='fallback + usda'")
        return (
            f"[source: fallback + usda{details}]\n\n"
            f"Top matches:\n{context}"
        )

    @staticmethod
    def _split_compare_items(query: str) -> list[str]:
        parts = []
        for chunk in query.replace(" versus ", " vs ").replace(" and ", " or ").split(" or "):
            item = chunk.replace(" vs ", " ").strip()
            if item and item not in parts:
                parts.append(item)
        return parts[:4]

    async def _search_item_for_compare(self, item_query: str) -> tuple[str, list[FoodProduct], str, str, int | None]:
        found, source, err, status = await self._search_usda(item_query, page_size=6)
        return item_query, found, source, err, status

    async def _search_usda(self, query: str, page_size: int) -> tuple[list[FoodProduct], str, str, int | None]:
        client = USDAFoodDataClient()
        found = await client.search_products(query, page_size=page_size)
        return found, "usda", client.last_error, client.last_status

    @staticmethod
    def _filter_relevant_products(query: str, products: list[FoodProduct]) -> tuple[list[FoodProduct], dict[str, str]]:
        sane_products = [p for p in products if AssistantService._is_reasonable_product(p)]
        if not sane_products:
            sane_products = products[:]
        if not sane_products:
            return [], {"confidence": "low", "explanation": "no relevant product match"}

        q = (query or "").lower()
        raw_tokens = [t for t in re.findall(r"[a-z0-9]+", q) if len(t) >= 3]
        q_tokens = []
        for tok in raw_tokens:
            q_tokens.extend(AssistantService._token_variants(tok))
        q_tokens = list(dict.fromkeys(q_tokens))
        if not q_tokens:
            return sane_products[:3], {"confidence": "low", "explanation": "query tokens too broad for strict filtering"}

        scored: list[tuple[int, FoodProduct]] = []
        for p in sane_products:
            hay = f"{p.product_name} {p.brands} {p.ingredients_text}".lower()
            score = 0
            if q in hay:
                score += 8
            for tok in q_tokens:
                if tok in hay:
                    score += 2
            # Keep exact-brand/product style matches first.
            if p.product_name.lower().startswith(q_tokens[0]):
                score += 1
            if score > 0:
                scored.append((score, p))

        if not scored:
            # For generic queries (e.g., apples/oranges), keep top raw results instead of emptying out.
            return sane_products[:3], {"confidence": "low", "explanation": "fallback to broad USDA search results"}

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score = scored[0][0]
        min_score = max(2, best_score - 3)
        scored = [x for x in scored if x[0] >= min_score]
        confidence = "high" if best_score >= 8 else "medium" if best_score >= 4 else "low"
        explanation = "exact keyword match" if best_score >= 8 else "partial keyword match"
        return [p for _, p in scored], {"confidence": confidence, "explanation": explanation}

    @staticmethod
    def _token_variants(token: str) -> list[str]:
        variants = [token]
        if token.endswith("es") and len(token) > 4:
            variants.append(token[:-2])
        if token.endswith("s") and len(token) > 3:
            variants.append(token[:-1])
        return variants

    @staticmethod
    def _infer_goal(text: str, session_state: dict[str, object]) -> str:
        lowered = (text or "").lower()
        rules = [
            ("lower calories", ["calorie", "kcal", "less calorie", "lower calorie", "calorie dense"]),
            ("lower sugar", ["sugar", "less sweet", "lower sugar"]),
            ("higher protein", ["protein", "more protein", "high protein"]),
            ("lower sodium", ["sodium", "salt", "electrolyte"]),
            ("lower fat", ["fat", "grease", "greasy"]),
        ]
        for goal, keywords in rules:
            if any(k in lowered for k in keywords):
                return goal
        prev = str(session_state.get("goal", "")).strip()
        return prev or "lower calories"

    @staticmethod
    def _metric_value(item: FoodProduct, goal: str) -> float:
        mapping = {
            "lower calories": item.energy_kcal_100g,
            "lower sugar": item.sugars_100g,
            "higher protein": item.proteins_100g,
            "lower sodium": item.salt_100g,
            "lower fat": item.fat_100g,
        }
        value = mapping.get(goal)
        if value is None:
            return float("-inf") if goal == "higher protein" else float("inf")
        return float(value)

    def _format_comparison_table(self, rows: list[tuple[str, FoodProduct]], goal: str) -> str:
        if not rows:
            return ""
        reverse = goal == "higher protein"
        ranked = sorted(rows, key=lambda x: self._metric_value(x[1], goal), reverse=reverse)
        lines = [
            "Comparison table",
            "",
            "| Query | Product | kcal/100g | sugar/100g | protein/100g | fat/100g | salt/100g |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for query, item in ranked:
            lines.append(
                f"| {query} | {item.product_name} | {self._fmt_num(item.energy_kcal_100g)} | "
                f"{self._fmt_num(item.sugars_100g)} | {self._fmt_num(item.proteins_100g)} | "
                f"{self._fmt_num(item.fat_100g)} | {self._fmt_num(item.salt_100g)} |"
            )
        lines.append("")
        lines.append(f"Assumed goal: {goal}")
        return "\n".join(lines)

    @staticmethod
    def _fmt_num(value: float | None) -> str:
        if value is None:
            return "n/a"
        return f"{value:.1f}"

    @staticmethod
    def _is_reasonable_product(item: FoodProduct) -> bool:
        # USDA occasionally returns malformed/odd entries with unrealistic energy values.
        # Keep broad ranges to avoid dropping valid foods while filtering clear outliers.
        if item.energy_kcal_100g is None:
            return True
        return 0.0 <= float(item.energy_kcal_100g) <= 900.0

    def _build_session_state(self, history: list[dict] | None) -> dict[str, object]:
        products = self._recall_product_queries(history)
        goal = ""
        if history:
            for msg in reversed(history):
                if str(msg.get("role", "")).lower() != "user":
                    continue
                text = str(msg.get("content", "")).strip()
                if not text:
                    continue
                goal = self._infer_goal(text, {"goal": ""})
                if goal:
                    break
        return {"products": products, "goal": goal or "lower calories"}

    def _ensure_natural_answer(
        self,
        answer: str,
        rows: list[tuple[str, FoodProduct]],
        goal: str,
        is_compare: bool,
    ) -> str:
        text = (answer or "").strip()
        if text and not text.startswith("OpenAI request failed") and not text.startswith("I couldn't generate"):
            return text
        if not rows:
            return "I could not generate a detailed recommendation. I found limited catalog data for your request."

        if is_compare and len(rows) >= 2:
            reverse = goal == "higher protein"
            ranked = sorted(rows, key=lambda x: self._metric_value(x[1], goal), reverse=reverse)
            best_query, best_item = ranked[0]
            second_query, second_item = ranked[1]
            return (
                f"For the goal '{goal}', the better match appears to be **{best_item.product_name}** "
                f"(query: {best_query}) over **{second_item.product_name}** (query: {second_query}). "
                "See the table below for the numeric breakdown and data gaps."
            )

        query, item = rows[0]
        return (
            f"Here is the nutrition summary I found for **{item.product_name}** (query: {query}). "
            f"Given your goal '{goal}', review calories, sugar, protein, fat, and salt in the table below."
        )

    @staticmethod
    def _session_state_text(session_state: dict[str, object]) -> str:
        products = ", ".join(session_state.get("products", [])) or "none"
        goal = str(session_state.get("goal", "lower calories"))
        return f"products={products}; goal={goal}"

    @staticmethod
    def _needs_goal_clarification(text: str) -> bool:
        lowered = (text or "").lower()
        asks_better = any(x in lowered for x in ["better", "healthier", "best option", "which is best"])
        has_goal = any(
            x in lowered
            for x in ["calorie", "kcal", "sugar", "protein", "sodium", "salt", "fat", "electrolyte"]
        )
        return asks_better and not has_goal

    @staticmethod
    def _needs_disambiguation(query: str, products: list[FoodProduct]) -> bool:
        if len(products) < 2:
            return False
        names = []
        for p in products[:4]:
            n = p.product_name.strip().lower()
            if n and n not in names:
                names.append(n)
        query_words = len((query or "").split())
        # Ask to clarify only for short/generic queries, not descriptive branded ones.
        return len(names) >= 3 and query_words <= 2

    @staticmethod
    def _should_force_catalog_mode(text: str) -> bool:
        lowered = (text or "").lower()
        catalog_cues = ["show me", "find me", "look up", "nutrition facts for", "product", "products", "option", "options"]
        return any(cue in lowered for cue in catalog_cues)

    def _recall_product_queries(self, history: list[dict] | None) -> list[str]:
        if not history:
            return []
        found: list[str] = []
        seen = set()
        for msg in history:
            if str(msg.get("role", "")).lower() != "user":
                continue
            text = str(msg.get("content", "")).strip()
            if not text:
                continue
            extracted = self.chat.extract_food_query(text, use_history=False)
            mode = str(extracted.get("mode", "")).strip().lower()
            if mode == "catalog":
                query = str(extracted.get("food_query", "")).strip()
                if query and query not in seen:
                    seen.add(query)
                    found.append(query)
            elif mode == "compare":
                items = extracted.get("compare_items", []) or []
                for item in items:
                    q = str(item).strip()
                    if q and q not in seen:
                        seen.add(q)
                        found.append(q)
        return found[-8:]

    def _latest_actionable_user_query(self, history: list[dict] | None) -> str:
        if not history:
            return ""
        for msg in reversed(history):
            if str(msg.get("role", "")).lower() != "user":
                continue
            text = str(msg.get("content", "")).strip()
            if not text:
                continue
            extracted = self.chat.extract_food_query(text, use_history=False)
            mode = str(extracted.get("mode", "")).lower()
            if mode in {"memory", "correction"}:
                continue
            return text
        return ""
