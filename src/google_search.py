"""
Google 可编程搜索（Custom Search JSON API）客户端封装。

本模块对 Google 自定义搜索 JSON API 做了轻量封装，用于以编程方式进行网页检索。
默认从环境变量获取凭据：

必须设置的环境变量：
- GOOGLE_API_KEY: Google API Key
- GOOGLE_CSE_ID: 可编程搜索引擎 ID（又名 cx，等价环境变量名为 GOOGLE_CX）

参考文档：
https://developers.google.com/custom-search/v1/overview
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

import requests
from requests.adapters import HTTPAdapter

try:
    # urllib3 is a transitive dependency of requests
    from urllib3.util.retry import Retry  # type: ignore
except Exception:  # pragma: no cover - very unlikely not to exist with requests
    Retry = None  # type: ignore

try:
    # Optional: load .env if available to simplify local development
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    # dotenv is optional; ignore if not installed
    pass


GOOGLE_CSE_ENDPOINT = "https://www.googleapis.com/customsearch/v1"


class GoogleSearchError(Exception):
    """Raised when the Google Search API returns an error or an invalid response."""


class GoogleSearchConfigError(GoogleSearchError):
    """Raised when API key or CSE ID are missing or invalid."""


@dataclass
class SearchResultItem:
    title: str
    link: str
    snippet: str
    display_link: Optional[str] = None
    formatted_url: Optional[str] = None


@dataclass
class SearchResponse:
    total_results: int
    search_time: float
    items: List[SearchResultItem]
    raw: Dict[str, Any]


def _build_session(timeout: float = 15.0, retries: int = 3, backoff_factor: float = 0.5) -> requests.Session:
    """创建带重试策略的 requests Session。

    参数：
        timeout: 每次请求的默认超时（秒）。
        retries: 短暂性错误的自动重试次数。
        backoff_factor: 指数退避倍数（控制重试间隔）。

    返回：
        配置完成的 requests.Session 实例。
    """
    session = requests.Session()
    # 为所有请求注入默认超时，避免无休止挂起
    session.request = _with_default_timeout(session.request, timeout)  # type: ignore
    if Retry is not None:
        retry = Retry(
            total=retries,
            read=retries,
            connect=retries,
            status=retries,
            backoff_factor=backoff_factor,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET", "POST"),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
    return session


def _with_default_timeout(request_func, timeout: float):
    """包装 session.request：若未显式给出 timeout，则注入默认超时。"""

    def wrapper(method, url, **kwargs):  # type: ignore[no-untyped-def]
        if "timeout" not in kwargs:
            kwargs["timeout"] = timeout
        return request_func(method, url, **kwargs)

    return wrapper


class GoogleSearchClient:
    """Google 自定义搜索 JSON API 的客户端。

    使用示例：
        client = GoogleSearchClient()
        resp = client.search("site:example.com test", num=5)
        for item in resp.items:
            print(item.title, item.link)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        cse_id: Optional[str] = None,
        session: Optional[requests.Session] = None,
        endpoint: str = GOOGLE_CSE_ENDPOINT,
    ) -> None:
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.cse_id = cse_id or os.getenv("GOOGLE_CSE_ID") or os.getenv("GOOGLE_CX")
        self.endpoint = endpoint
        self.session = session or _build_session()

        if not self.api_key or not self.cse_id:
            raise GoogleSearchConfigError(
                "Missing Google API credentials. Set GOOGLE_API_KEY and GOOGLE_CSE_ID (or GOOGLE_CX)."
            )

    def search(
        self,
        query: str,
        *,
        num: int = 10,
        start: int = 1,
        site: Optional[str] = None,
        exact_terms: Optional[str] = None,
        exclude_terms: Optional[str] = None,
        language: Optional[str] = None,
        country: Optional[str] = None,
        safe: str = "off",
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> SearchResponse:
        """执行单页搜索。

        参数：
            query: 搜索关键词。
            num: 本页返回结果条数，范围 [1, 10]（Google 限制）。
            start: 起始结果位置，1 为基准（1-based）。
            site: 限定站点/域名，例如 "example.com"。
            exact_terms: 强制包含的精确短语。
            exclude_terms: 需要排除的词语。
            language: 语言限制，例如 "lang_en"。
            country: 地区/国家代码（影响地理偏置），例如 "us"。
            safe: 安全搜索："off" 或 "active"。
            extra_params: 传递给 API 的其它原始参数。

        返回：
            SearchResponse，包含解析后的字段和原始响应。
        """
        if not 1 <= num <= 10:
            raise ValueError("num must be between 1 and 10 per Google API limits")
        if start < 1:
            raise ValueError("start must be >= 1 (1-based index)")

        params: Dict[str, Any] = {
            "q": query,
            "key": self.api_key,
            "cx": self.cse_id,
            "num": num,
            "start": start,
            "safe": safe,
        }
        if site:
            params["siteSearch"] = site
        if exact_terms:
            params["exactTerms"] = exact_terms
        if exclude_terms:
            params["excludeTerms"] = exclude_terms
        if language:
            params["lr"] = language  # e.g., "lang_en"
        if country:
            params["gl"] = country  # e.g., "us"
        if extra_params:
            params.update(extra_params)

        # 发送请求
        resp = self.session.get(self.endpoint, params=params)
        if resp.status_code != 200:
            raise GoogleSearchError(
                f"Google API error: HTTP {resp.status_code} - {resp.text[:300]}"
            )
        data: Dict[str, Any] = resp.json()
        if "error" in data:
            # Structured API error
            err = data["error"].get("message", "Unknown error")
            raise GoogleSearchError(f"Google API error: {err}")

        items_raw = data.get("items", []) or []
        # 解析结果项
        items = [
            SearchResultItem(
                title=i.get("title", ""),
                link=i.get("link", ""),
                snippet=i.get("snippet", ""),
                display_link=i.get("displayLink"),
                formatted_url=i.get("formattedUrl"),
            )
            for i in items_raw
        ]

        search_info = data.get("searchInformation", {})
        total_results = int(search_info.get("totalResults", "0") or 0)
        search_time = float(search_info.get("searchTime", 0.0) or 0.0)

        return SearchResponse(
            total_results=total_results,
            search_time=search_time,
            items=items,
            raw=data,
        )

    def iter_results(
        self,
        query: str,
        *,
        max_results: int = 20,
        site: Optional[str] = None,
        exact_terms: Optional[str] = None,
        exclude_terms: Optional[str] = None,
        language: Optional[str] = None,
        country: Optional[str] = None,
        safe: str = "off",
        page_size: int = 10,
    ) -> Iterator[SearchResultItem]:
        """跨页迭代返回结果，直到达到 max_results。

        注意：Google 限制单页最多 10 条；总配额与账号的 API 配额策略相关。
        """
        yielded = 0
        start = 1
        while yielded < max_results:
            batch_size = min(page_size, max_results - yielded)
            resp = self.search(
                query,
                num=batch_size,
                start=start,
                site=site,
                exact_terms=exact_terms,
                exclude_terms=exclude_terms,
                language=language,
                country=country,
                safe=safe,
            )
            if not resp.items:
                break
            for item in resp.items:
                yield item
                yielded += 1
                if yielded >= max_results:
                    break
            start += batch_size
