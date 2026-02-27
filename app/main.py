import gradio as gr

from app.config import settings
from app.services.assistant_service import AssistantService

service = AssistantService()


async def chat_fn(message: str, history: list[dict]) -> str:
    return await service.answer(message, history=history)


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="OpenCommerce AI Assistant") as demo:
        gr.Markdown(
            """
            # Nutrition Assistant (USDA + LLM)
            Compare products, inspect nutrition facts, and recall what you asked earlier in the session.
            Start with your goal if you have one: lower calories, lower sugar, higher protein, or lower sodium.
            """
        )
        gr.ChatInterface(
            fn=chat_fn,
            type="messages",
            examples=[
                "Hello",
                "Compare Snickers and Kit Kat to see which is less calorie dense",
                "Can you tell me the nutrition facts for a Monster energy drink?",
                "List the products I asked about earlier",
            ],
        )
    return demo


if __name__ == "__main__":
    app = build_demo()
    app.launch(server_name=settings.gradio_server_name, server_port=settings.gradio_server_port)
