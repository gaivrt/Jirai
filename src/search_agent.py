"""
Search Query Rewrite Agent

使用 LLM 将用户提供的关键词/查询改写成更适合 Web 搜索的多个查询变体。

特性：
- 可配置最大变体数、温度、提示词模板
- 解析 LLM 返回的 JSON（尽量健壮）
- 若 LLM 不可用，回退为原始输入（可选）
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .llm_analyzer import LLMUnavailableError


def _load_aisuite():
    try:
        import aisuite as ai  # type: ignore
        return ai
    except Exception as e:  # pragma: no cover
        raise LLMUnavailableError(
            "未安装或无法导入 aisuite。请安装并正确配置，"
            "或将 search_agent 切换到你偏好的 LLM 客户端。"
        ) from e


@dataclass
class QueryRewriteConfig:
    model: str
    temperature: float = 0.3
    max_variants: int = 5
    prompt_template_path: Optional[str] = None
    max_prompt_chars: int = 4000


class SearchQueryAgent:
    def __init__(self, *, config: QueryRewriteConfig) -> None:
        self._ai = _load_aisuite()
        self._client = self._ai.Client()
        self.config = config
        self._last_log: Optional[Dict[str, Any]] = None

    def rewrite_queries(
        self,
        seed: str,
        *,
        language_hint: Optional[str] = None,
        country_hint: Optional[str] = None,
        site: Optional[str] = None,
    ) -> List[str]:
        prompt = self._build_prompt(
            seed=seed,
            language_hint=language_hint,
            country_hint=country_hint,
            site=site,
            max_variants=self.config.max_variants,
        )
        if len(prompt) > self.config.max_prompt_chars:
            prompt = prompt[: self.config.max_prompt_chars]

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a web search query rewrite agent. Return only JSON with a 'queries' array."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        resp = self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
        )
        content = resp.choices[0].message.content  # type: ignore[no-any-return]
        queries = _parse_queries_from_json_text(content)

        # 记录日志（输入/输出）以便上层落盘
        self._last_log = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_variants": self.config.max_variants,
            "prompt_template_path": self.config.prompt_template_path,
            "request": {
                "messages": messages,
            },
            "response": {
                "raw": content,
                "parsed_queries": queries,
            },
        }
        return queries

    def _build_prompt(
        self,
        *,
        seed: str,
        language_hint: Optional[str],
        country_hint: Optional[str],
        site: Optional[str],
        max_variants: int,
    ) -> str:
        template_text = _load_prompt_template(
            self.config.prompt_template_path or "prompts/search_query_rewrite.md"
        )
        return template_text.format(
            keywords=seed,
            language=language_hint or "",
            country=country_hint or "",
            site=site or "",
            max_variants=max_variants,
        )

    def get_last_log(self) -> Optional[Dict[str, Any]]:
        return self._last_log


def _load_prompt_template(path: str) -> str:
    p = Path(path)
    if p.exists():
        return p.read_text(encoding="utf-8")
    # 兜底的极简模板
    return (
        "Given the user keywords: {keywords}\n"
        "Rewrite them into up to {max_variants} focused web search queries.\n"
        "Prefer concise queries that retrieve high-signal pages.\n"
        "If provided, consider language: {language}, country: {country}, and site: {site}.\n"
        "Return strictly as JSON with a single key 'queries': an array of strings."
    )


def _parse_queries_from_json_text(text: str) -> List[str]:
    """从 LLM 的文本中解析出 {"queries": [...]}，尽量健壮。"""
    # 优先解析代码块 ```json ... ```
    code_fences = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    candidates: List[str] = []
    if code_fences:
        candidates.extend(code_fences)
    # 尝试全文解析
    candidates.append(text)

    for cand in candidates:
        cand_strip = cand.strip()
        # 尝试找到第一对大括号
        brace_match = re.search(r"\{[\s\S]*\}", cand_strip)
        if brace_match:
            json_str = brace_match.group(0)
        else:
            json_str = cand_strip
        try:
            data = json.loads(json_str)
            q = data.get("queries")
            if isinstance(q, list):
                # 去重、裁剪、清洗
                out: List[str] = []
                for s in q:
                    if not isinstance(s, str):
                        continue
                    s2 = s.strip()
                    if s2 and s2 not in out:
                        out.append(s2)
                if out:
                    return out
        except Exception:
            continue

    # 最后兜底：按行拆分，取形如 "- xxx" 或普通行
    lines = [ln.strip("- •\t ") for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    # 只取看起来像查询的短行
    out: List[str] = []
    for ln in lines:
        if len(ln) > 200:
            continue
        if ln and ln not in out:
            out.append(ln)
    return out[:10]
