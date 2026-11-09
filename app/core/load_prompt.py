from pathlib import Path
from app.utils.path_util import PROJECT_ROOT
from app.core.logger import logger  # 可选，加日志更友好

def load_prompt(name: str, **kwargs) -> str:
    prompt_path = PROJECT_ROOT / 'prompts' / f'{name}.prompt'

    if not prompt_path.exists():
        raise FileNotFoundError(f"提示词文件不存在：{prompt_path.absolute()}")

    raw_prompt = prompt_path.read_text(encoding='utf-8')

    if kwargs:
        rendered_prompt = raw_prompt.format(**kwargs)
        logger.debug(f"提示词渲染成功，替换变量：{list(kwargs.keys())}")
        return rendered_prompt
    return raw_prompt



if __name__ == '__main__':
    root_folder = "hl3070使用说明书"  # 要替换的文件名称
    image_content = ("这是图片的上文内容", "这是图片的下文内容")  # 要替换的上下文
    final_prompt = load_prompt(
        name='image_summary',
        root_folder=root_folder,  # 对应{root_folder}
        image_content=image_content  # 对应{image_content[0]}、{image_content[1]}
    )
    print("✅ 渲染后的最终提示词：")
    print(final_prompt)