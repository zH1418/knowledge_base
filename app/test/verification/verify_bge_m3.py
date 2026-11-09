from app.core.logger import logger
from app.lm.embedding_utils import get_bge_m3_ef, generate_embeddings


def verify():
    logger.info("正在加载 BGE-M3 模型...")
    model = get_bge_m3_ef()
    logger.success("BGE-M3 模型加载成功")

    texts = ["测试商品名称", "苏伯尔5000W大功率电磁炉"]
    logger.info(f"正在为 {len(texts)} 条文本生成向量...")
    result = generate_embeddings(texts)

    dense_list = result["dense"]
    sparse_list = result["sparse"]

    print(f"\n  PASS - 稠密向量: {len(dense_list)} 条，每条 {len(dense_list[0])} 维")
    print(f"  PASS - 稀疏向量: {len(sparse_list)} 条")
    print(f"  PASS - 稀疏向量样本: {sparse_list[0]}")


if __name__ == "__main__":
    print("\n============================================================")
    print("  BGE-M3 模型验证")
    print("============================================================")
    verify()
    print("============================================================\n")
