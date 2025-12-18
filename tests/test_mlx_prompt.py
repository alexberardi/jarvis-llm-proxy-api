import base64

from backends.mlx_backend import MlxClient
from managers.chat_types import ImagePart, NormalizedMessage, TextPart


def test_build_prompt_and_images_orders_parts():
    # 1x1 transparent PNG
    img_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="
    )
    messages = [
        NormalizedMessage(
            role="user",
            content=[
                TextPart(text="describe"),
                ImagePart(data=img_bytes, mime_type="image/png"),
                TextPart(text="please"),
            ],
        )
    ]

    prompt, images = MlxClient._build_prompt_and_images(messages, tokenizer=None)

    assert "<image>" in prompt
    assert "describe" in prompt and "please" in prompt
    assert prompt.index("describe") < prompt.index("<image>")
    assert len(images) == 1

