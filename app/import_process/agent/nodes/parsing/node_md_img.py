import hashlib
import json
import os
import re
from pathlib import Path
from typing import Optional
from urllib.parse import quote

from app.core.exceptions import InvalidStateError, ParameterError
from app.core.logger import logger
from minio import Minio
from minio.deleteobjects import DeleteObject

from app.conf.minio_config import minio_config
from app.import_process.agent.node_base import NodeBase
from app.import_process.agent.state import ImportGraphState
from app.lm.vlm_utils import batch_summarize
from app.import_process.utils.minio_utils import get_minio_client


class NodeMdImg(NodeBase):

    name = "Markdown图片处理"

    _SUPPORTED_EXTS = frozenset(
        {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    )
    _CONTEXT_LEN: int = 100
    _CONTEXT_EXTEND: int = 50       # 边界搜索最大延伸距离
    _CACHE_FILE: str = "_summaries_cache.json"
    _SENTENCE_BOUNDARY: frozenset[str] = frozenset("。！？!?\n")


    @staticmethod
    def _compile_image_pattern(image_file: str) -> re.Pattern:
        """编译用于匹配 MD 中指定图片引用的正则表达式。"""
        return re.compile(r"!\[.*?\]\(.*?" + re.escape(image_file) + r".*?\)")

    @staticmethod
    def _file_md5(file_path: str) -> str:
        """计算文件的 MD5 哈希值。"""
        return hashlib.md5(Path(file_path).read_bytes()).hexdigest()


    def process(self, state: ImportGraphState) -> ImportGraphState:
        md_content, md_path_obj, images_dir = self._get_md_context(state)
        state["md_content"] = md_content

        if images_dir is None:
            return state  # 无图片目录，直接跳过

        target_images = self._scan_images(md_content, images_dir)
        if not target_images:
            logger.info("未检测到 MD 中引用的支持格式图片，跳过处理")
            return state

        cache = self._load_cache(images_dir)
        summaries: dict[str, str] = {}
        uncached: list[tuple[str, str, tuple[str, str], str]] = []

        for img_file, img_path, context in target_images:
            md5 = self._file_md5(img_path)
            if md5 in cache:
                summaries[img_file] = cache[md5]
                logger.info(f"摘要缓存命中：{img_file}")
            else:
                uncached.append((img_file, img_path, context, md5))

        if uncached:
            new_summaries = batch_summarize(
                [(fn, fp, ctx) for fn, fp, ctx, _ in uncached],
                md_path_obj.stem,
            )
            for (img_file, _, _, md5), summary in zip(
                    uncached, [new_summaries[fn] for fn, _, _, _ in uncached]
            ):
                summaries[img_file] = summary
                cache[md5] = summary
            self._save_cache(images_dir, cache)

        new_md_content = self._upload_and_replace_md(
            md_path_obj.stem, target_images, summaries, md_content
        )

        new_md_path = self._save_processed_md(state["md_path"], new_md_content)
        state["md_content"] = new_md_content
        state["md_path"] = new_md_path
        return state


    def _get_md_context(
        self, state: ImportGraphState
    ) -> tuple[str, Path, Optional[Path]]:
        md_path = state.get("md_path", "").strip()
        if not md_path:
            raise ParameterError("缺失参数：md_path")
        md_path_obj = Path(md_path)
        if not md_path_obj.exists():
            raise InvalidStateError(f"MD文件不存在，绝对路径: {md_path_obj.absolute()}")

        if state.get("md_content"):
            md_content = state["md_content"]
        else:
            with open(md_path_obj, "r", encoding="utf-8") as f:
                md_content = f.read()
        images_dir = md_path_obj.parent / "images"
        if not images_dir.exists():
            images_dir = None

        return md_content, md_path_obj, images_dir


    def _scan_images(
        self, md_content: str, images_dir: Path
    ) -> list[tuple[str, str, tuple[str, str]]]:
        target_images = []

        for image_file in os.listdir(images_dir):
            ext = os.path.splitext(image_file)[1].lower()
            if ext not in self._SUPPORTED_EXTS:
                continue

            img_path = str(images_dir / image_file)

            context = self._find_image_context(md_content, image_file)
            if not context:
                logger.warning(f"图片未在 MD 中引用，跳过：{image_file}")
                continue

            target_images.append((image_file, img_path, context))

        logger.info(f"扫描到 {len(target_images)} 张待处理图片")

        return target_images

    def _find_image_context(
        self, md_content: str, image_file: str
    ) -> Optional[tuple[str, str]]:
        pattern = self._compile_image_pattern(image_file)
        match = pattern.search(md_content)
        if not match:
            return None

        start, end = match.span()
        text_len = len(md_content)

        pre_start = max(0, start - self._CONTEXT_LEN)
        if pre_start > 0 and md_content[pre_start - 1] not in self._SENTENCE_BOUNDARY:
            search_limit = max(0, pre_start - self._CONTEXT_EXTEND)
            for i in range(pre_start - 1, search_limit - 1, -1):
                if md_content[i] in self._SENTENCE_BOUNDARY:
                    pre_start = i + 1
                    break
        pre = md_content[pre_start:start]

        post_end = min(text_len, end + self._CONTEXT_LEN)
        if post_end < text_len and md_content[post_end] not in self._SENTENCE_BOUNDARY:
            search_limit = min(text_len, post_end + self._CONTEXT_EXTEND)
            for i in range(post_end, search_limit):
                if md_content[i] in self._SENTENCE_BOUNDARY:
                    post_end = i + 1
                    break
        post = md_content[end:post_end]

        return pre, post


    def _load_cache(self, images_dir: Path) -> dict[str, str]:
        """从缓存文件加载 MD5→摘要 映射。"""
        cache_path = images_dir / self._CACHE_FILE
        if not cache_path.exists():
            return {}
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"读取缓存失败，将重建缓存：{e}")
            return {}

    def _save_cache(self, images_dir: Path, cache: dict[str, str]) -> None:
        """原子保存 MD5→摘要 映射到缓存文件（先写临时文件，再 rename）。"""
        cache_path = images_dir / self._CACHE_FILE
        temp_path = cache_path.with_suffix(".tmp")
        try:
            temp_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(temp_path, cache_path)  # 原子操作
        except OSError as e:
            logger.error(f"保存缓存失败：{e}")


    def _upload_and_replace_md(
        self,
        doc_stem: str,
        target_images: list[tuple[str, str, tuple[str, str]]],
        summaries: dict[str, str],
        md_content: str,
    ) -> str:
        """将图片上传到 MinIO，并替换 MD 中的本地引用为远程 URL + 摘要。"""
        minio_client = get_minio_client()
        upload_dir = f"{minio_config.minio_img_dir}/{doc_stem}/"

        urls = self._batch_upload(minio_client, upload_dir, target_images)

        image_info: dict[str, tuple[str, str]] = {}
        for img_file, summary in summaries.items():
            url = urls.get(img_file)
            if url:
                image_info[img_file] = (summary, url)

        new_md = self._replace_md_images(md_content, image_info)

        if image_info:
            self._clean_minio_dir(minio_client, upload_dir)
        else:
            logger.error("所有图片上传失败，已跳过 MinIO 目录清理，保留原有远端文件")

        return new_md

    def _clean_minio_dir(self, minio_client: Minio, prefix: str) -> None:
        """删除 MinIO 指定前缀下的所有对象。"""
        try:
            objects = list(minio_client.list_objects(
                bucket_name=minio_config.bucket_name,
                prefix=prefix,
                recursive=True,
            ))
            if objects:
                delete_list = [DeleteObject(obj.object_name) for obj in objects]
                errors = minio_client.remove_objects(minio_config.bucket_name, delete_list)
                for error in errors:  # ← 惰性迭代器必须遍历才执行删除
                    logger.error(f"MinIO 删除失败：{error}")
                logger.info(f"MinIO 目录已清理：{prefix}，{len(objects)} 个对象")
        except Exception as e:
            logger.error(f"MinIO 目录清理异常：{e}")

    def _batch_upload(
        self,
        minio_client: Minio,
        upload_dir: str,
        target_images: list[tuple[str, str, tuple[str, str]]],
    ) -> dict[str, str]:
        """批量上传图片到 MinIO，返回 {文件名: URL}。"""
        urls: dict[str, str] = {}
        for img_file, img_path, _ in target_images:
            object_name = upload_dir + img_file
            try:
                minio_client.fput_object(
                    bucket_name=minio_config.bucket_name,
                    object_name=object_name,
                    file_path=img_path,
                )
                obj_encoded = quote(object_name, safe="/")
                protocol = "https" if minio_config.minio_secure else "http"
                base = f"{protocol}://{minio_config.endpoint}/{minio_config.bucket_name}"
                urls[img_file] = f"{base}/{obj_encoded}"
                size_kb = os.path.getsize(img_path) / 1024
                logger.info(f"上传成功：{img_file}（{size_kb:.1f}KB）→ {urls[img_file]}")
            except Exception as e:
                logger.error(f"上传失败：{img_file}，{e}")
        return urls

    def _replace_md_images(
        self, md_content: str, image_info: dict[str, tuple[str, str]]
    ) -> str:
        """使用摘要和 URL 替换 MD 中的图片引用。"""
        for img_file, (summary, url) in image_info.items():
            pattern = self._compile_image_pattern(img_file)
            md_content = pattern.sub(lambda m: f"![{summary}]({url})", md_content)
        return md_content


    def _save_processed_md(self, origin_md_path: str, new_content: str) -> str:
        """将处理后的 MD 内容保存为 *_new.md，返回新路径。"""
        new_path = os.path.splitext(origin_md_path)[0] + "_new.md"
        with open(new_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        logger.info(f"处理后 MD 已保存：{new_path}")
        return new_path
