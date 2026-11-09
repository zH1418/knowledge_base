#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np

try:
    from app.utils.normalize_sparse_vector import normalize_sparse_vector
except ImportError:
    def normalize_sparse_vector(sparse_vec: dict[int, float] | None) -> dict[int, float]:
        if not sparse_vec:
            return {}
        values = np.array(list(sparse_vec.values()), dtype=np.float64)
        l2_norm = np.linalg.norm(values)
        if l2_norm < 1e-9:
            return sparse_vec
        normalized = (values / l2_norm).tolist()
        return {dim: float(val) for (dim, _), val in zip(sparse_vec.items(), normalized)}


def print_vector_info(label: str, vec: dict[int, float] | None) -> None:
    """打印向量内容及其 L2 范数"""
    if vec is None:
        vec = {}
    print(f"{label}: {vec}")
    if vec:
        values = np.array(list(vec.values()), dtype=np.float64)
        norm = np.linalg.norm(values)
        print(f"  范数 (L2): {norm:.8f}")
    else:
        print("  范数 (L2): 0.0")


def test_normalize_sparse_vector() -> None:
    """执行多项测试用例，打印归一化前后对比"""
    test_cases = [
        ("标准向量 (3,4) => 归一化为 (0.6,0.8)", {3: 3.0, 5: 4.0}),
        ("空字典", {}),
        ("None 输入", None),
        ("全零向量 (极小值)", {1: 0.0, 2: 0.0, 3: 0.0}),
        ("已归一化的向量 (范数已为1)", {10: 0.6, 20: 0.8}),
        ("含负值的向量", {0: -2.0, 3: 2.0, 5: -1.0}),
        ("高维稀疏向量", {100: 1.0, 200: 2.0, 300: 3.0, 400: 4.0, 500: 5.0}),
    ]

    print("=" * 70)
    print("稀疏向量 L2 归一化测试")
    print("=" * 70)

    for idx, (desc, vec) in enumerate(test_cases, 1):
        print(f"\n[测试 {idx}] {desc}")
        print("-" * 50)
        print_vector_info("  原始向量", vec)

        normalized = normalize_sparse_vector(vec)
        print_vector_info("  归一化后", normalized)

        if normalized:
            vals = np.array(list(normalized.values()), dtype=np.float64)
            norm_after = np.linalg.norm(vals)
            if abs(norm_after - 1.0) < 1e-6:
                print("  ✅ 归一化后范数 ≈ 1.0，验证通过")
            else:
                print(f"  ⚠️ 归一化后范数 = {norm_after:.8f}，偏离1.0（可能因浮点误差）")
        print("-" * 50)

    print("\n所有测试用例执行完毕。")


def demo_generate_embeddings_mock() -> None:
    print("\n" + "=" * 70)
    print("模拟 BGE-M3 生成混合向量后的稀疏归一化演示")
    print("=" * 70)

    mock_sparse_list = [
        {10: 0.5, 20: 1.2, 30: 0.3},
        {5: 2.0, 15: 1.0},
        {},
    ]

    print("原始稀疏向量列表（模型原始输出）:")
    for i, sp in enumerate(mock_sparse_list):
        print(f"  文档 {i+1}: {sp}")

    print("\n应用归一化后:")
    normalized_list = [normalize_sparse_vector(sp) for sp in mock_sparse_list]
    for i, sp_norm in enumerate(normalized_list):
        print(f"  文档 {i+1}: {sp_norm}")
        if sp_norm:
            vals = np.array(list(sp_norm.values()))
            print(f"    范数: {np.linalg.norm(vals):.8f}")

    print("\n（注：实际使用中，generate_embeddings 会自动调用 normalize_sparse_vector。）")


if __name__ == "__main__":
    test_normalize_sparse_vector()
    demo_generate_embeddings_mock()