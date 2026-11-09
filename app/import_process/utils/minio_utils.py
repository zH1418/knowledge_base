import json
import os
from typing import Optional
from minio import Minio
from app.conf.minio_config import minio_config
from app.core.logger import logger
from app.core.decorators import log_execution, safe_execute
from app.core.exceptions import ObjectStorageException, ConfigurationError

_minio_client: Optional[Minio] = None
_bucket_name: Optional[str] = None


def get_minio_client() -> Minio:
    global _minio_client, _bucket_name

    if _minio_client is not None:
        return _minio_client

    endpoint = minio_config.endpoint
    access_key = minio_config.access_key
    secret_key = minio_config.secret_key
    bucket_name = minio_config.bucket_name

    missing = []
    if not endpoint:
        missing.append("MINIO_ENDPOINT")
    if not access_key:
        missing.append("MINIO_ACCESS_KEY")
    if not secret_key:
        missing.append("MINIO_SECRET_KEY")

    if missing:
        raise ConfigurationError(
            message="MinIO 配置不完整",
            detail={"missing_configs": missing},
        )

    try:
        _minio_client = Minio(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=os.getenv("MINIO_SECURE", "false").lower() == "true",
        )
        logger.info(f"[MinIO] 客户端初始化成功 | endpoint={endpoint}")

        if bucket_name and not _minio_client.bucket_exists(bucket_name):
            logger.info(f"[MinIO] 存储桶[{bucket_name}]不存在，开始创建")
            _minio_client.make_bucket(bucket_name)
            logger.info(f"[MinIO] 存储桶[{bucket_name}]创建成功")
        elif bucket_name:
            logger.info(f"[MinIO] 存储桶[{bucket_name}]已存在，无需重复创建")
        else:
            bucket_name = minio_config.bucket_name or "knowledge-base-files"
            if not _minio_client.bucket_exists(bucket_name):
                _minio_client.make_bucket(bucket_name)

        bucket_policy = {
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"AWS": ["*"]},
                "Action": ["s3:GetObject"],
                "Resource": [f"arn:aws:s3:::{bucket_name}/*"]
            }]
        }
        _minio_client.set_bucket_policy(bucket_name, json.dumps(bucket_policy))
        logger.info(f"[MinIO] 存储桶[{bucket_name}]已配置公网只读策略")

        _bucket_name = bucket_name
        return _minio_client

    except (ConfigurationError, ObjectStorageException):
        raise
    except Exception as e:
        raise ObjectStorageException(
            message=f"MinIO 客户端初始化失败: {e}",
            original=e,
        ) from e


@log_execution(component="minio")
@safe_execute(component="minio", default_return=None, reraise=False)
def upload_file(object_name: str, file_path: str, bucket_name: str = None) -> Optional[str]:
    client = get_minio_client()
    bucket = bucket_name or minio_config.bucket_name
    client.fput_object(bucket, object_name, file_path)
    logger.info(f"[MinIO] 文件上传成功 | bucket={bucket} | object={object_name}")
    return object_name


@log_execution(component="minio")
@safe_execute(component="minio", default_return=None, reraise=False)
def download_file(object_name: str, file_path: str, bucket_name: str = None) -> Optional[str]:
    client = get_minio_client()
    bucket = bucket_name or minio_config.bucket_name
    client.fget_object(bucket, object_name, file_path)
    logger.info(f"[MinIO] 文件下载成功 | bucket={bucket} | object={object_name} -> {file_path}")
    return file_path


if __name__ == "__main__":
    try:
        client = get_minio_client()
        logger.success("MinIO 测试：客户端获取成功")
    except Exception as e:
        logger.error(f"MinIO 测试失败：{e}")
