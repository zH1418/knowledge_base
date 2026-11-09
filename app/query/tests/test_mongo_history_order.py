import time

from app.clients.mongo_history_utils import (
    save_chat_message, get_recent_messages, clear_history,
)

SID = f"test_{int(time.time())}"


def test_1_recent_n():
    """验证 limit=3 时返回的是最新的 3 条，非最旧的 3 条。"""
    print("\n=== test_1: get_recent_messages 返回最近的 N 条 ===")

    conv = [
        ("user",  "第1轮-问: 万用表怎么测电压？"),
        ("assistant", "第1轮-答: 选择直流电压档..."),
        ("user",  "第2轮-问: 那个的精度呢？"),
        ("assistant", "第2轮-答: 精度通常为 0.5%~2%..."),
        ("user",  "第3轮-问: 示波器怎么调波形？"),
        ("assistant", "第3轮-答: 连接探头到CH1..."),
        ("user",  "第4轮-问: 它和万用表哪个好？"),
        ("assistant", "第4轮-答: 各有优势，示波器侧重波形..."),
    ]
    for role, text in conv:
        save_chat_message(SID, role, text)

    msgs = get_recent_messages(SID, limit=3)
    print(f"  limit=3，返回 {len(msgs)} 条:")
    for m in msgs:
        print(f"    [{m['role']}] {m['text']}")

    assert len(msgs) == 3
    assert "第3轮" in msgs[0]["text"], f"第1条应含'第3轮'，实际: {msgs[0]['text'][:20]}"
    assert "第4轮" in msgs[2]["text"], f"最后条应含'第4轮'，实际: {msgs[2]['text'][:20]}"
    print("  ✅ 返回的是最新消息，非最旧")


def test_2_ascending_order():
    """验证返回的消息按时间正序排列（旧→新）。"""
    print("\n=== test_2: 时间正序验证 ===")

    conv = [
        ("user",  "最早的消息"),
        ("assistant", "最早的回复"),
        ("user",  "最新的消息"),
        ("assistant", "最新的回复"),
    ]
    for role, text in conv:
        save_chat_message(SID, role, text)

    msgs = get_recent_messages(SID, limit=10)
    print(f"  返回 {len(msgs)} 条:")
    for m in msgs:
        print(f"    [{m['role']}] {m['text'][:30]}")

    ts_list = [m["ts"] for m in msgs]
    assert all(ts_list[i] <= ts_list[i+1] for i in range(len(ts_list)-1)), \
        f"ts 未递增: {ts_list}"
    assert msgs[0]["text"] == "最早的消息", f"第一条消息不对: {msgs[0]['text']}"
    assert msgs[-1]["text"] == "最新的回复", f"最后条消息不对: {msgs[-1]['text']}"
    print("  ✅ 时间正序（旧→新），ts 递增")


def test_3_format_output():
    """模拟 _format_history_text 的格式化输出。"""
    print("\n=== test_3: _format_history_text 模拟 ===")

    messages = get_recent_messages(SID, limit=6)
    lines = []
    for msg in messages[-6:]:
        label = "用户" if msg.get("role") == "user" else "助手"
        lines.append(f"{label}: {msg['text']}")

    formatted = "\n".join(lines)
    print("  格式化后的 prompt 输入:\n  ---")
    for line in lines:
        print(f"  {line}")
    print("  ---")

    assert lines[0].startswith("用户:"), f"首行应为用户: {lines[0][:20]}"
    assert lines[-1].startswith("助手:"), f"末行应为助手: {lines[-1][:20]}"
    print("  ✅ 顺序: 用户→助手→用户→助手")


if __name__ == "__main__":
    try:
        test_1_recent_n()
        clear_history(SID)  # 清理 test_1 数据
        test_2_ascending_order()
        test_3_format_output()
    finally:
        clear_history(SID)
        print("\n全部测试完成 ✅")
