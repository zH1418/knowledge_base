import json, time, os, threading
import requests

V1 = "http://127.0.0.1:8002"
V2 = "http://127.0.0.1:8003"
DESKTOP = os.path.expanduser("~/Desktop")

TESTS = [
    ("闲聊",    "你好"),
    ("枚举标准", "有哪些产品"),
    ("枚举口语", "你都有什么设备"),
    ("精确搜索", "HAK180烫金机怎么设置温度"),
    ("精确搜索2","万用表怎么测量电压"),
    ("模糊搜索", "烫金机"),
    ("图片检索", "HAK180烫金机有哪些安全警示"),
    ("错别字",   "万用标怎么用"),
    ("KB未命中", "iPhone 16 Pro Max 怎么开机"),
    ("综合查询", "万用表和烫金机哪个更贵"),
]

_lock = threading.Lock()


def test_v1(name: str, query: str) -> dict:
    t0 = time.perf_counter()
    try:
        r = requests.post(f"{V1}/query", json={
            "query": query, "session_id": f"v1-{hash(query)&0xFFFF}", "is_stream": False,
        }, timeout=180)
        data = r.json()
        return {"ok": True, "time_s": round(time.perf_counter() - t0, 1),
                "answer": data.get("answer","") or "", "type": data.get("query_type",""),
                "items": data.get("item_names",[]), "sources": len(data.get("sources",[]) or [])}
    except Exception as e:
        return {"ok": False, "time_s": 0, "answer": f"ERROR: {e}"}


def test_v2(name: str, query: str) -> dict:
    t0 = time.perf_counter()
    full = ""
    try:
        r = requests.post(f"{V2}/query/stream", json={"question": query}, stream=True, timeout=180)
        for line in r.iter_lines():
            if not line: continue
            line = line.decode("utf-8", errors="ignore")
            if line.startswith("data: "):
                try: evt = json.loads(line[6:])
                except: continue
                if evt.get("type") == "token":
                    full += evt.get("content", "")
        return {"ok": True, "time_s": round(time.perf_counter() - t0, 1), "answer": full}
    except Exception as e:
        return {"ok": False, "time_s": 0, "answer": f"ERROR: {e}"}


HEADER_V1 = "# V1 (LangGraph) 检索测试结果\n\n**模型**: deepseek-v4-pro-260425\n**时间**: {ts}\n\n---\n\n"
HEADER_V2 = "# V2 (FC Agent) 检索测试结果\n\n**模型**: deepseek-v4-pro-260425\n**时间**: {ts}\n\n---\n\n"


def append_v1(name, query, r):
    with _lock:
        with open(os.path.join(DESKTOP, "V1_LangGraph_测试结果.md"), "a", encoding="utf-8") as f:
            f.write(f"## {name}: {query}\n\n")
            f.write(f"- **耗时**: {r['time_s']}s\n")
            f.write(f"- **类型**: {r.get('type','')} | **实体**: {r.get('items',[])} | **来源**: {r.get('sources',0)}\n\n")
            f.write("### 回答\n\n")
            f.write(r["answer"])
            f.write("\n\n---\n\n")


def append_v2(name, query, r):
    with _lock:
        with open(os.path.join(DESKTOP, "V2_FC_Agent_测试��果.md"), "a", encoding="utf-8") as f:
            f.write(f"## {name}: {query}\n\n")
            f.write(f"- **耗时**: {r['time_s']}s\n\n")
            f.write("### 回答\n\n")
            f.write(r["answer"])
            f.write("\n\n---\n\n")


def run_v1():
    for name, query in TESTS:
        r = test_v1(name, query)
        append_v1(name, query, r)
        print(f"[V1] {name}: {r['time_s']}s ({len(r['answer'])} chars)")


def run_v2():
    for name, query in TESTS:
        r = test_v2(name, query)
        append_v2(name, query, r)
        print(f"[V2] {name}: {r['time_s']}s ({len(r['answer'])} chars)")


def main():
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    for path, header in [
        ("V1_LangGraph_测试结果.md", HEADER_V1.format(ts=ts)),
        ("V2_FC_Agent_测试结果.md", HEADER_V2.format(ts=ts)),
    ]:
        with open(os.path.join(DESKTOP, path), "w", encoding="utf-8") as f:
            f.write(header)

    print(f"V1 vs V2 并发测试 ({len(TESTS)} 条)，逐条写入桌面")
    print("=" * 50)

    t1 = threading.Thread(target=run_v1, name="V1", daemon=True)
    t2 = threading.Thread(target=run_v2, name="V2", daemon=True)

    t0 = time.perf_counter()
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    elapsed = time.perf_counter() - t0

    print(f"\n测试完成，总��时 {elapsed:.1f}s")
    print(f"桌面: V1_LangGraph_测试结果.md / V2_FC_Agent_测试结果.md")


if __name__ == "__main__":
    main()
