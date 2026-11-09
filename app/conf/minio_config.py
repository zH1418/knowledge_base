from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class MinIOConfig:
    endpoint: str    # MinIO服务地址（含http/https和端口）
    access_key: str  # MinIO访问密钥（对应MINIO_ACCESS_KEY）
    secret_key: str  # MinIO秘钥（对应MINIO_SECRET_KEY）
    bucket_name: str # MinIO默认存储桶名（知识库文件专用）
    minio_img_dir: str #Minio存储图片的文件夹
    minio_secure: bool # 是否使用ssl加密 http 还是 https


minio_config = MinIOConfig(
    endpoint=os.getenv("MINIO_ENDPOINT", ""),
    access_key=os.getenv("MINIO_ACCESS_KEY", ""),
    secret_key=os.getenv("MINIO_SECRET_KEY", ""),
    bucket_name=os.getenv("MINIO_BUCKET_NAME", "knowledge-base-files"),
    minio_img_dir=os.getenv("MINIO_IMG_DIR", "/upload-images"),
    minio_secure=os.getenv("MINIO_SECURE", "False") == "True",
)