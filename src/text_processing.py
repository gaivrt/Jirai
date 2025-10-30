"""
网页抓取与文本处理工具。

功能包含：
- 抓取网页并提取正文（移除脚本/样式/导航等非正文）
- 归一化空白符（便于后续分块与输入 LLM）
- 简单的按字符长度分块（带重叠），快速实用
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter

try:
    from urllib3.util.retry import Retry  # type: ignore
except Exception:  # pragma: no cover
    Retry = None  # type: ignore


DEFAULT_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0 Safari/537.36 JiraiBot/0.1"
)


def _build_session(timeout: float = 15.0, retries: int = 2) -> requests.Session:
    session = requests.Session()
    session.request = _with_default_timeout(session.request, timeout)  # type: ignore
    if Retry is not None:
        retry = Retry(
            total=retries,
            read=retries,
            connect=retries,
            status=retries,
            backoff_factor=0.4,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET", "HEAD"),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
    return session


def _with_default_timeout(request_func, timeout: float):
    def wrapper(method, url, **kwargs):  # type: ignore[no-untyped-def]
        if "timeout" not in kwargs:
            kwargs["timeout"] = timeout
        return request_func(method, url, **kwargs)

    return wrapper


@dataclass
class FetchedDocument:
    url: str
    title: str
    text: str


def fetch_and_extract(
    url: str,
    *,
    user_agent: str = DEFAULT_UA,
    timeout: float = 15.0,
    max_bytes: int = 1_572_864,  # ~1.5 MiB 上限，避免巨型页面阻塞
) -> Optional[FetchedDocument]:
    """抓取单个 URL 并用 BeautifulSoup 提取可见文本。

    当 Content-Type 非 HTML 或遇到严重错误时返回 None。
    """
    session = _build_session(timeout=timeout)
    headers = {"User-Agent": user_agent, "Accept": "text/html,application/xhtml+xml"}
    try:
        # 直接 GET，但限制读取体积，确保在网络不佳或页面超大时尽快返回
        resp = session.get(
            url,
            headers=headers,
            allow_redirects=True,
            stream=True,
            timeout=(max(2.0, timeout / 2), timeout),  # 明确设置连接/读取超时
        )
    except Exception:
        return None

    ctype = resp.headers.get("Content-Type", "").lower()
    if "text/html" not in ctype and "application/xhtml" not in ctype:
        return None

    # 读取有限的正文字节以避免卡死在大文件
    buf = bytearray()
    try:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                buf.extend(chunk)
                if len(buf) >= max_bytes:
                    break
    except Exception:
        return None

    # 尝试推断编码并解码
    encoding = resp.encoding or getattr(resp, "apparent_encoding", None) or "utf-8"
    try:
        html = buf.decode(encoding, errors="replace")
    except Exception:
        try:
            html = buf.decode("utf-8", errors="replace")
        except Exception:
            return None
    soup = BeautifulSoup(html, "html.parser")
    # 移除非正文元素，减少噪音
    for tag in soup(["script", "style", "noscript", "iframe", "form", "header", "footer", "nav"]):
        tag.decompose()

    # 提取标题（若存在）
    title = soup.title.string.strip() if soup.title and soup.title.string else ""

    # 提取并清洗正文文本
    text = soup.get_text("\n")
    text = _normalize_whitespace(text)
    if not text or len(text) < 100:
        return None

    return FetchedDocument(url=url, title=title, text=text)


def _normalize_whitespace(text: str) -> str:
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[\t\f\v]", " ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def chunk_text(text: str, *, size: int = 2000, overlap: int = 200) -> List[str]:
    """按字符长度进行滑动分块（带重叠）。简单且高效。

    参数：
        text: 输入文本。
        size: 目标分块大小（字符数）。
        overlap: 相邻分块的重叠字符数。
    """
    if size <= 0:
        return [text]
    if overlap >= size:
        overlap = max(0, size // 5)
    chunks: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        end = min(n, i + size)
        chunks.append(text[i:end])
        if end >= n:
            break
        i = end - overlap
    return chunks
