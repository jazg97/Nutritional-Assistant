---
title: Nutrition Assistant Bot
emoji: "🥗"
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "5.49.1"
python_version: "3.11"
app_file: app.py
pinned: false
---

# Nutrition Assistant Bot

An open-source nutrition assistant using USDA FoodData Central and LLM reasoning.

## Current mode

The app runs in **nutrition assistant mode** with USDA FoodData Central + OpenAI.

## Day 1 scope

- Gradio chat app
- Live product retrieval from USDA FoodData Central
- LLM answer generation over retrieved nutrition context
- Fallback output when LLM/API is unavailable

## Why this project

This repo is designed to demonstrate:

- LLM application design with structured JSON outputs
- Real-time data integration with a public API
- Retrieval and ranking for recommendation-style answers
- Production-minded code organization

## Architecture

- `app/main.py`: Gradio entrypoint
- `app/services/assistant_service.py`: request orchestration
- `app/data_providers/usda.py`: USDA FoodData Central API client
- `app/llm/responder.py`: LLM chat and context-based answering
- `notebooks/nutrition_assistant_evaluation.ipynb`: evaluation notebook
- `evaluation/eval_cases.jsonl`: labeled evaluation prompts

## Quick start

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy env template:

```bash
cp .env.example .env
```

4. Run:

```bash
python -m app.main
```

## Environment variables

- `OPENAI_API_KEY`: enables LLM responses.
- `OPENAI_BASE_URL`: optional, for OpenAI-compatible providers.
- `OPENAI_MODEL`: optional, default model.
- `DEBUG_LOG`: set `1` to print request/response debug traces in CLI.
- `USDA_API_KEY`: FoodData Central API key.
- `USDA_PAGE_SIZE`: USDA results per search.
- `REQUEST_TIMEOUT_SECONDS`: API timeout.
- `GRADIO_SERVER_NAME`: default `0.0.0.0`.
- `GRADIO_SERVER_PORT`: default `7860`.

## Next steps (Day 2+)

- Add vector embeddings for semantic retrieval over nutrition labels
- Add evaluation notebook (retrieval quality, response grounding, latency)
- Add WhatsApp channel adapter
- Add CI workflow and richer test coverage
