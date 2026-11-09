import requests

BASE = "http://localhost:8002"
TESTS = [
    ("闲聊你好", "你好", "chitchat",
     ["chitchat 判为闲聊不硬编码", "LLM 自由应答"]),
    ("闲聊谢谢", "谢谢", "chitchat",
     ["问候变体也识别为闲聊"]),
    ("枚举-标准", "有哪些产品", "enumeration",
     ["产品名列表", "无 rewrite"]),
    ("枚举-口语1", "你的库里有啥", "enumeration",
     ["口语化枚举语义理解"]),
    ("枚举-口语2", "你都知道些什么", "enumeration",
     ["口语化枚举变体"]),
    ("精确搜索", "HAK180烫金机怎么设置温度", "search",
     ["温度参数", "110-130℃"]),
    ("模糊搜索", "烫金机", "search",
     ["HyDE 扩展", "HAK180"]),
    ("错别字纠正", "万用标的使用方法", "search",
     ["纠正为万用表", "DT9205A"]),
    ("口语搜索", "那个烫金的东西怎么用", "search",
     ["指代推断为烫金机"]),
    ("KB未命中", "iPhone 16 Pro Max 怎么开机", "search",
     ["web 兜底", "开机"]),
    ("非产品类", "今天有哪些科技新闻", "unknown",
     ["走联网搜索", "不通检索"]),
    ("上下文追问", "你上次说了什么", "chitchat",
     ["历史上下文", "区分于 search"]),
]

print("=" * 60)
print("  查询管线全链路测试 v2")
print("=" * 60)

passed = 0
failed = 0

for name, query, expected_type, checks in TESTS:
    print(f"\n--- {name}: {query} ---")
    try:
        r = requests.post(f"{BASE}/query", json={
            "query": query,
            "session_id": f"e2ev2-{name}",
            "is_stream": False,
        }, timeout=180)
        resp = r.json()
        answer = resp.get("answer", "") or ""
        item_names = resp.get("item_names", []) or []
        query_type = resp.get("query_type", "")

        ok = r.status_code == 200
        type_match = query_type == expected_type
        status = "✅" if ok and type_match else ("⚠️" if ok else "❌")

        print(f"  HTTP {r.status_code} | type={query_type} (want {expected_type}) {status}")
        if not type_match:
            print(f"  ⚠ 预期 type={expected_type} 实际={query_type}")
        print(f"  items: {item_names}")
        print(f"  answer ({len(answer)} chars): {answer[:120]}...")

        for check in checks:
            found = check in answer or check.lower() in answer.lower()
            if not found:
                print(f"  ⚠ 检查项未命中: {check}")
        if ok and type_match:
            passed += 1
        else:
            failed += 1
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        failed += 1

print(f"\n{'='*60}")
print(f"  通过 {passed}/{passed+failed}")
if failed:
    print(f"  失败 {failed} 项需排查")
print("=" * 60)
