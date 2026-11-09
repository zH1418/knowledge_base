import os

from modelscope.hub.snapshot_download import snapshot_download

cache_dir = os.environ.get(
    "MODELSCOPE_CACHE", os.path.join("models", "modelscope_cache")
)
model_dir = snapshot_download('Xorbits/bge-m3', cache_dir=cache_dir)
print(f"模型已下载到: {model_dir}")
