import numpy as np

from app.core.logger import logger
from app.lm.embedding_utils import generate_embeddings, get_bge_m3_ef


def check(desc: str, condition: bool) -> str:
    return "  PASS" if condition else "  FAIL"


def run_test(name: str, test_fn) -> None:
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    try:
        test_fn()
    except Exception as e:
        print(f"  异常: {type(e).__name__}: {e}")



def test_model_singleton():
    ef1 = get_bge_m3_ef()
    ef2 = get_bge_m3_ef()
    print(check("模型单例可用", ef1 is not None))
    print(check("两次获取返回同一实例", ef1 is ef2))



def test_dense_normalization():
    result = generate_embeddings(["苏伯尔5000W电磁炉", "智能温控技术参数"])
    dense = result["dense"]

    print(check("生成 2 个稠密向量", len(dense) == 2))
    print(check("稠密维度 = 1024", len(dense[0]) == 1024))

    for i, vec in enumerate(dense):
        vec_np = np.array(vec, dtype=np.float64)
        norm = np.linalg.norm(vec_np)
        ok = abs(norm - 1.0) < 0.01
        print(check(f"dense[{i}] L2={norm:.6f} ≈ 1.0", ok))



def test_sparse_normalization():
    result = generate_embeddings(["苏伯尔5000W大功率电磁炉", "额定功率5000W电压220V"])
    sparse = result["sparse"]

    print(check("生成 2 个稀疏向量", len(sparse) == 2))

    for i, sv in enumerate(sparse):
        vals = np.array(list(sv.values()), dtype=np.float64)
        norm = np.linalg.norm(vals)
        ok = abs(norm - 1.0) < 0.01
        print(check(f"sparse[{i}]: {len(sv)} dims, L2={norm:.6f} ≈ 1.0", ok))

        all_float = all(isinstance(v, float) and not isinstance(v, np.floating) for v in sv.values())
        print(check(f"sparse[{i}]: 值均为 Python float", all_float))



def test_value_types():
    result = generate_embeddings(["测试"])
    dense = result["dense"][0]
    sparse = result["sparse"][0]

    print(check("dense 元素为 Python float", all(isinstance(v, float) for v in dense)))
    if sparse:
        k_type = type(next(iter(sparse)))
        v_type = type(next(iter(sparse.values())))
        print(check(f"sparse key 为 int", k_type is int))
        print(check(f"sparse value 为 float", v_type is float and not issubclass(v_type, np.floating)))



def test_batch_consistency():
    text1 = "苏伯尔电磁炉"
    text2 = "美的微波炉"
    r1 = generate_embeddings([text1])
    r2 = generate_embeddings([text2])
    r_batch = generate_embeddings([text1, text2])

    d1_batch_diff = max(abs(a - b) for a, b in zip(r1["dense"][0], r_batch["dense"][0]))
    d2_batch_diff = max(abs(a - b) for a, b in zip(r2["dense"][0], r_batch["dense"][1]))
    print(check("text1 单条/批量 dense 一致", d1_batch_diff < 1e-6))
    print(check("text2 单条/批量 dense 一致", d2_batch_diff < 1e-6))



def test_empty_input():
    from app.core.exceptions import ParameterError, KBException

    try:
        generate_embeddings([])
        print(check("空列表应抛异常", False))
    except (ParameterError, ValueError, KBException):
        print(check("空列表 → 异常", True))


def test_single_text():
    result = generate_embeddings(["单文本"])
    print(check("单文本 dense 非空", len(result["dense"]) == 1))
    print(check("单文本 sparse 非空", len(result["sparse"]) == 1))



def test_text_length_impact():
    short = generate_embeddings(["电磁炉"])
    long_ = generate_embeddings(["苏伯尔5000W大功率电磁炉智能温控多档火力调节安全保护功能"])
    print(check("短文本 dense 维度 = 1024", len(short["dense"][0]) == 1024))
    print(check("长文本 dense 维度 = 1024", len(long_["dense"][0]) == 1024))
    print(check("长短文本 sparse 维度不同", len(short["sparse"][0]) != len(long_["sparse"][0])))



if __name__ == "__main__":
    logger.info("===== BGE-M3 向量生成与归一化测试 =====")

    run_test("1. 模型单例加载", test_model_singleton)
    run_test("2. 稠密向量归一化", test_dense_normalization)
    run_test("3. 稀疏向量归一化", test_sparse_normalization)
    run_test("4. 值类型验证", test_value_types)
    run_test("5. 单条/批量一致性", test_batch_consistency)
    run_test("6.1 空输入边界", test_empty_input)
    run_test("6.2 单文本输入", test_single_text)
    run_test("7. 长短文本影响", test_text_length_impact)

    logger.info("===== BGE-M3 向量生成与归一化测试结束 =====")
