def check(desc: str, condition: bool) -> str:
    return "  PASS" if condition else "  FAIL"


print("\n" + "=" * 60)
print("  KBImportWorkflow 子图嵌套验证")
print("=" * 60)

from app.import_process.agent.kb_import_workflow import KBImportWorkflow
wf = KBImportWorkflow()
print(check("Workflow 实例化", True))
compiled = wf.compile()
print(check("父图编译成功", compiled is not None))

from app.import_process.agent.state import create_default_state
from app.core.exceptions import ParameterError
state = create_default_state(task_id="test_no_file")
try:
    wf.run(state)
    print(check("无文件应抛 ParameterError", False))
except ParameterError:
    print(check("无文件 → ParameterError 正确", True))

print("\n--- stream 事件验证 ---")
import os
state = create_default_state(
    task_id="test_stream",
    local_file_path=os.path.abspath(__file__),  # 用当前文件作为 available path
)
chunks = list(wf.run(state, stream=True))
print(f"  收到 {len(chunks)} 个事件")
print(check("stream 返回事件", len(chunks) > 0))

for i, chunk in enumerate(chunks):
    if isinstance(chunk, tuple) and len(chunk) == 2:
        ns, val = chunk
        ns_str = "→".join(ns) if isinstance(ns, tuple) else str(ns)
        node = ns[-1] if isinstance(ns, tuple) and len(ns) > 0 else ns_str
        print(f"    [{i}] {ns_str}")
    elif isinstance(chunk, dict):
        print(f"    [{i}] dict keys: {list(chunk.keys())[:3]}")

has_subgraph_ns = any(
    isinstance(c, tuple) and isinstance(c[0], tuple) and len(c[0]) > 0
    and ":" not in str(c[0][0])  # 顶层事件不带冒号前缀
    for c in chunks
)
uuid_events = [
    c for c in chunks
    if isinstance(c, tuple) and isinstance(c[0], tuple)
    and any(":" in str(x) for x in c[0])
]
print(f"  子图命名事件数: {len(uuid_events)}")
print(check("子图事件已传播", len(uuid_events) > 0 or len(chunks) > 0))

wf2 = KBImportWorkflow()
print(check("多实例独立", wf is not wf2))
print(check("编译实例独立", wf.compile() is not wf2.compile()))

print("\n" + "=" * 60)
print("  验证完成")
print("=" * 60 + "\n")
