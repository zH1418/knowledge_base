import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()



MODEL = ("qwen-turbo"
         "")

api_key = os.getenv("MCP_WEBSEARCH_API_KEY")
if not api_key:
    raise RuntimeError("MCP_WEBSEARCH_API_KEY 未设置")

client = OpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=api_key,
)

question = input("请输入测试问题（回车使用默认）: ").strip()
if not question:
    question = "请用100字介绍北京。"

print(f"\n平台: 阿里百炼 (DashScope)")
print(f"模型: {MODEL}")
print(f"问题: {question}")
print("=" * 60)

try:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": question}],
        max_tokens=2048,
    )

    answer = response.choices[0].message.content
    print(f"回答 ({len(answer)} 字符):")
    print(answer)
    print(f"\n使用 tokens: {response.usage}")

except Exception as e:
    print(f"请求失败: {type(e).__name__}: {e}")
