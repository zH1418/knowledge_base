import base64
import os
from collections import deque

from langchain_core.messages import HumanMessage

from app.conf.lm_config import lm_config
from app.core.decorators import log_execution
from app.core.load_prompt import load_prompt
from app.core.logger import logger
from app.lm.lm_utils import get_llm_client
from app.utils.rate_limit_utils import apply_api_rate_limit

_VLM_REQUEST_DEQUE = deque()


@log_execution(component="vlm")
def summarize_image(image_path: str, root_folder: str, image_context: tuple[str, str]) -> str:
    prompt_text = load_prompt(
        name="image_summary",
        root_folder=root_folder,  # → prompts/image_summary.prompt 的 {root_folder}
        image_content=image_context,  # → {image_content[0]}、{image_content[1]}
    )

    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")

    image_ext = os.path.splitext(image_path)[1][1:]  # 如 "jpg", "png"
    messages = [
        HumanMessage(content=[
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": f"data:image/{image_ext};base64,{base64_image}"}},
        ])
    ]

    lvm_client = get_llm_client(model=lm_config.lv_model)

    response = lvm_client.invoke(messages)
    summary = response.content.strip().replace("\n", "")
    return summary

@log_execution(component="vlm")
def batch_summarize(target_images, doc_stem) -> dict[str, str]:
    summaries = {}

    for img_file, img_path, context in target_images:
        apply_api_rate_limit(_VLM_REQUEST_DEQUE, max_requests=10, window_seconds=30)

        try:
            summaries[img_file] = summarize_image(
                image_path=img_path,
                root_folder=doc_stem,
                image_context=context,
            )
            logger.info(f"图片摘要完成：{img_file}")
        except Exception as e:
            logger.error(f"图片摘要失败：{img_file}，{e}（已跳过，不阻塞后续图片）")

    return summaries  # {"图片名": "摘要文字", ...}