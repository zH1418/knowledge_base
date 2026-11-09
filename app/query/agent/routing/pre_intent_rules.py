from __future__ import annotations

import re

_CHITCHAT_WORDS: set[str] = {
    "你好", "您好", "hi", "hello", "嗨", "哈喽", "哈啰", "在吗", "在么", "有人吗",
    "谢谢", "多谢", "感谢", "谢谢你", "谢谢啦", "thanks", "thankyou",
    "再见", "拜拜", "bye", "goodbye", "晚安", "早上好", "中午好", "下午好", "晚上好",
    "你是谁", "你叫什么", "你是谁呀", "你好呀", "您好呀",
}

_ENUM_PATTERNS: list[re.Pattern] = [
    re.compile(r"有(哪些|什么|啥)(产品|型号|设备|商品|东西|分类|品类)?"),
    re.compile(r"都(有|是)(哪些|什么|啥|些什么)"),
    re.compile(r"(列(一下|出来|出来呗)?|列举|罗列)(一下)?"),
    re.compile(r"库里(有啥|有什么|有哪些)"),
    re.compile(r"你(都)?(会|能|可以)(做)?(些什么|什么|啥|干什么)"),
    re.compile(r"你(都)?知道(些)?(什么|啥)"),
    re.compile(r"知识库(里)?(有啥|有什么|有哪些)"),
]

_RETRIEVAL_INTENT: re.Pattern = re.compile(
    r"怎么|如何|为什么|咋|怎样|步骤|方法|故障|维修|参数|对比|区别|设置|安装|使用"
)

_PUNCT: re.Pattern = re.compile(r"[\s，。！？!?、,.~～…\-—]+")


def classify_early(query: str) -> tuple[str, str] | None:
    q = (query or "").strip()
    norm = _PUNCT.sub("", q).lower()

    if not norm:
        return ("", "您好，请描述您想了解的产品信息。")

    if norm in _CHITCHAT_WORDS or (len(norm) <= 6 and norm in _CHITCHAT_WORDS):
        return ("chitchat", "")

    if not _RETRIEVAL_INTENT.search(q):
        for pat in _ENUM_PATTERNS:
            if pat.search(q):
                return ("enumeration", "")

    return None
