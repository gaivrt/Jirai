"""
基于 LLM 的分析器：将已抓取与处理的文本整理为 Alignment Report（对齐报告）。

默认使用 `aisuite` 进行推理调用，与现有示例（`aisuit_demo.py`）保持一致。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional


class LLMUnavailableError(Exception):
    pass


@dataclass
class EvidenceChunk:
    source_url: str
    source_title: str
    text: str


def _load_aisuite():
    try:
        import aisuite as ai  # type: ignore
        return ai
    except Exception as e:  # pragma: no cover
        raise LLMUnavailableError(
            "未安装或无法导入 aisuite。请安装并正确配置，"
            "或改造 LLMAnalyzer 以使用你偏好的 LLM 客户端。"
        ) from e


class LLMAnalyzer:
    def __init__(
        self,
        *,
        model: str,
        temperature: float = 0.2,
        system_prompt: Optional[str] = None,
        max_input_chars: int = 12000,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.system_prompt = system_prompt or (
            "You are a careful, objective analyst. Write a concise, well-structured "
            "Alignment Report grounded strictly in the provided evidence."
        )
        self.max_input_chars = max_input_chars
        self._ai = _load_aisuite()
        self._client = self._ai.Client()
        self._last_log = None

    def analyze(
        self,
        chunks: Iterable[EvidenceChunk],
        *,
        prompt_template: str,
        title: str = "Alignment Report",
    ) -> str:
        """生成 Markdown 格式的 Alignment Report（对齐报告）。

        为避免超过模型输入限制，会将拼接后的证据信息在 `max_input_chars` 处截断。
        """
        rendered_evidence = self._render_evidence(chunks)
        original_evidence_len = len(rendered_evidence)
        truncated = False
        if original_evidence_len > self.max_input_chars:
            rendered_evidence = rendered_evidence[: self.max_input_chars]
            truncated = True

        user_content = prompt_template.format(
            title=title,
            evidence=rendered_evidence,
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        content = resp.choices[0].message.content  # type: ignore[no-any-return]

        # 保存本次调用的日志（输入/输出），供上层记录到磁盘
        self._last_log = {
            "model": self.model,
            "temperature": self.temperature,
            "max_input_chars": self.max_input_chars,
            "request": {
                "messages": messages,
                "evidence": {
                    "length_original": original_evidence_len,
                    "length_used": len(rendered_evidence),
                    "truncated": truncated,
                },
            },
            "response": {
                "content": content,
            },
        }
        return content

    def get_last_log(self):
        return self._last_log

    @staticmethod
    def _render_evidence(chunks: Iterable[EvidenceChunk]) -> str:
        lines: List[str] = []
        for i, c in enumerate(chunks, start=1):
            header = f"### Evidence {i}: {c.source_title or c.source_url}\n{c.source_url}"
            lines.append(header)
            lines.append("")
            lines.append(c.text)
            lines.append("")
        return "\n".join(lines)
