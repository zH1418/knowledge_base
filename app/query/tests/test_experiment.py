from app.lm.lm_utils import get_llm_client

MODEL = "doubao-seed-2-0-mini-260428"

print("=" * 60)
print("测试 1: factual 短问答")
print("=" * 60)
short_prompt = (
    "请根据以下信息回答问题，尽可能详细：\n\n"
    "【参考信息】\n"
    "HAK180 烫金机，温度范围 50-250℃，默认 120℃。压力分轻/中/重三档。\n"
    "时间 0.5-5 秒。每 5000 次涂抹高温润滑脂。\n\n"
    "【问题】\n"
    "HAK180烫金机怎么设置温度和压力？"
)
print(f"prompt_chars: {len(short_prompt)}")
llm = get_llm_client(model=MODEL)
resp = llm.invoke(short_prompt)
short_answer = str(resp.content)
print(f"answer_chars: {len(short_answer)}")
print(short_answer[:300])

print("\n" + "=" * 60)
print("测试 2: 长输出 —— 要求 2000 字科幻小说开篇")
print("=" * 60)
long_prompt = (
    "请写一篇科幻小说的开篇，至少 2000 字。\n\n"
    "背景设定：2079 年，人类在月球背面发现了一座外星文明遗留的\"回声塔\"——\n"
    "任何进入塔内的人，都能听到自己已故亲友的声音。主角林渡是一位失去女儿的地质学家，\n"
    "被派遣前往月球基地参与回声塔的研究。\n\n"
    "要求：\n"
    "1. 从林渡乘坐飞船抵达月球基地开始写起\n"
    "2. 描写月面场景、基地内部环境、人物心理\n"
    "3. 引入冲突或悬念（如回声塔出现异常信号）\n"
    "4. 语言需要有文学性，适当运用细节描写\n"
    "5. 总字数不少于 2000 字"
)
print(f"prompt_chars: {len(long_prompt)}")
resp2 = llm.invoke(long_prompt)
long_answer = str(resp2.content)
print(f"answer_chars: {len(long_answer)}")
print("=" * 60)
print(long_answer[:500])
if len(long_answer) > 500:
    print(f"\n... （共 {len(long_answer)} 字符）...")
    print(f"...末尾: ...{long_answer[-300:]}")
