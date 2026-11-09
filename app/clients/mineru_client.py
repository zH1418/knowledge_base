import shutil
import time
import zipfile
from pathlib import Path
from typing import Tuple

import requests
from loguru import logger

from app.conf.mineru_config import MineruConfig
from app.core.decorators import log_execution, retry_on_failure


class MineruClient:

    def __init__(self, config: MineruConfig):
        self.base_url = config.base_url.rstrip("/")
        self.api_token = config.api_token
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_token}",
        }

        if not self.base_url or not self.api_token:
            raise ValueError(
                "MinerU 配置缺失，请检查 .env 文件中的 "
                "MINERU_BASE_URL 和 MINERU_API_TOKEN"
            )


    @log_execution(component="mineru")
    @retry_on_failure(max_retries=3)
    def _get_upload_url(self, filename: str) -> Tuple[str, str]:
        url = f"{self.base_url}/file-urls/batch"
        payload = {
            "files": [{"name": filename}],
            "model_version": "vlm",
        }
        resp = requests.post(url, json=payload, headers=self.headers, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(
                f"获取上传链接失败：HTTP {resp.status_code}，响应：{resp.text}"
            )

        result = resp.json()
        if result.get("code") != 0:
            raise RuntimeError(f"获取上传链接业务失败：{result}")

        batch_id = result["data"]["batch_id"]
        signed_url = result["data"]["file_urls"][0]
        logger.info(f"上传链接获取成功，batch_id={batch_id}")
        return batch_id, signed_url

    def _upload_file(self, signed_url: str, file_path: Path) -> None:
        """将本地 PDF 上传至 MinerU 预签名 URL。"""
        logger.info(f"开始上传 PDF：{file_path.name}")
        with open(file_path, "rb") as f:
            resp = requests.put(signed_url, data=f, timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(
                f"文件上传失败：HTTP {resp.status_code}，响应：{resp.text}"
            )
        logger.info("PDF 上传成功")

    @log_execution(component="mineru")
    def _poll_result(self, batch_id: str, timeout: int = 600) -> str:
        url = f"{self.base_url}/extract-results/batch/{batch_id}"
        poll_interval = 3
        start = time.time()

        while True:
            elapsed = time.time() - start
            if elapsed > timeout:
                raise TimeoutError(
                    f"轮询超时（>{timeout}s），batch_id={batch_id}"
                )

            try:
                resp = requests.get(url, headers=self.headers, timeout=10)
            except Exception as e:
                logger.warning(f"轮询网络异常，{poll_interval}s 后重试：{e}")
                time.sleep(poll_interval)
                continue

            if resp.status_code != 200:
                raise RuntimeError(
                    f"轮询 HTTP 失败：{resp.status_code}，响应：{resp.text}"
                )

            data = resp.json()
            if data.get("code") != 0:
                raise RuntimeError(f"轮询业务错误：{data}")

            item = data["data"]["extract_result"][0]
            state = item["state"]

            if state == "done":
                logger.info(f"解析完成，耗时 {int(elapsed)}s")
                return item["full_zip_url"]

            if state == "failed":
                err_msg = item.get("err_msg", "未知错误")
                raise RuntimeError(f"解析失败，batch_id={batch_id}：{err_msg}")

            logger.info(
                f"解析中... 已耗时 {int(elapsed)}s，状态={state}"
            )
            time.sleep(poll_interval)

    def _download_and_extract(
        self, zip_url: str, output_dir: Path, doc_stem: str
    ) -> str:
        logger.info(f"下载 ZIP：{zip_url}")
        resp = requests.get(zip_url, timeout=120)
        if resp.status_code != 200:
            raise RuntimeError(
                f"ZIP 下载失败：HTTP {resp.status_code}，响应：{resp.text}"
            )

        zip_path = output_dir / f"{doc_stem}_result.zip"
        zip_path.write_bytes(resp.content)
        logger.info(f"ZIP 保存至：{zip_path}")

        extract_dir = output_dir / doc_stem
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        logger.info(f"ZIP 解压至：{extract_dir}")

        full_md = extract_dir / "full.md"
        if not full_md.exists():
            raise FileNotFoundError(f"解压后未找到 full.md，目录内容：{list(extract_dir.iterdir())}")

        new_md = full_md.with_name(f"{doc_stem}.md")
        full_md.rename(new_md)
        logger.info(f"MD 文件：{new_md.absolute()}")

        return str(new_md.absolute())


    @log_execution(component="mineru")
    def process_pdf(self, pdf_path: Path, output_dir: Path) -> str:
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF 文件不存在：{pdf_path}")
        output_dir.mkdir(parents=True, exist_ok=True)

        batch_id, signed_url = self._get_upload_url(pdf_path.name)
        self._upload_file(signed_url, pdf_path)
        zip_url = self._poll_result(batch_id)
        return self._download_and_extract(zip_url, output_dir, pdf_path.stem)
