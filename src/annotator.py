"""
Two-stage alignment-assisted annotation pipeline (Part B).

Stage 1 (Align/Explain):
- Input: raw text + alignment report (subculture terms/semantics) as context
- Output: JSON with modern rewrite of the text and identified subculture terms with explanations

Stage 2 (Judge):
- In the same conversation (retaining Stage 1 in history), ask the LLM to assign OD/ED/SH labels
  on an ordinal 0/1/2 scale, returning a strict JSON with labels and brief rationales.

This module wraps the LLM client via `aisuite` (same as Part A) and exposes a simple API for
running on single texts or datasets.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Optional progress bar
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore

import concurrent.futures


class LLMUnavailableError(Exception):
    pass


def _load_aisuite():
    try:
        import aisuite as ai  # type: ignore
        return ai
    except Exception as e:  # pragma: no cover
        raise LLMUnavailableError(
            "未安装或无法导入 aisuite。请安装并正确配置，或改造 Annotator 以使用你的 LLM 客户端。"
        ) from e


def _read_text_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return p.read_text(encoding="utf-8")


def _truncate(text: str, max_chars: int) -> Tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def _parse_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Robustly parse a JSON object from LLM output.

    - Prefer fenced ```json blocks
    - Else find first {...}
    - Else try the whole text
    """
    fences = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    candidates: List[str] = []
    if fences:
        candidates.extend(fences)
    candidates.append(text)
    for cand in candidates:
        s = cand.strip()
        m = re.search(r"\{[\s\S]*\}", s)
        js = m.group(0) if m else s
        try:
            data = json.loads(js)
            if isinstance(data, dict):
                return data
        except Exception:
            continue
    return None


def _md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def _inject(template: str, mapping: Dict[str, str]) -> str:
    """Safely inject {key} tokens by literal replacement without str.format semantics.

    This avoids KeyError for JSON braces present in the prompt template.
    """
    out = template
    for k, v in mapping.items():
        out = out.replace("{" + k + "}", v)
    return out


def _truncate_deep(obj: Any, *, max_chars: int = 2000, _seen: Optional[set] = None) -> Any:
    """Recursively truncate long strings inside nested structures to keep logs compact."""
    if _seen is None:
        _seen = set()
    try:
        oid = id(obj)
        if oid in _seen:
            return "<recursion>"
        _seen.add(oid)
    except Exception:
        pass

    if isinstance(obj, str):
        if len(obj) <= max_chars:
            return obj
        return obj[:max_chars] + f"\n...<truncated {len(obj) - max_chars} chars>"
    if isinstance(obj, list):
        return [_truncate_deep(x, max_chars=max_chars, _seen=_seen) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_truncate_deep(x, max_chars=max_chars, _seen=_seen) for x in obj)
    if isinstance(obj, dict):
        return {k: _truncate_deep(v, max_chars=max_chars, _seen=_seen) for k, v in obj.items()}
    return obj


@dataclass
class Stage1Result:
    modern_rewrite: str
    terms: List[Dict[str, Any]]
    detected_language: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None


@dataclass
class Stage2Result:
    labels: Dict[str, int]  # keys: OD, ED, SH with values 0/1/2
    rationale: Dict[str, str]
    raw: Optional[Dict[str, Any]] = None


class LLMClient:
    """Minimal aisuite chat client wrapper that preserves the last call log."""

    def __init__(self) -> None:
        self._ai = _load_aisuite()
        self._client = self._ai.Client()
        self._last_log: Optional[Dict[str, Any]] = None

    def chat(self, *, model: str, temperature: float, messages: List[Dict[str, str]], timeout: Optional[float] = None, max_retries: int = 2) -> str:
        import time
        attempts = max(0, int(max_retries))
        for attempt in range(attempts + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    timeout=timeout,
                )
                content = resp.choices[0].message.content  # type: ignore[no-any-return]
                self._last_log = {
                    "model": model,
                    "temperature": temperature,
                    "messages": messages,
                    "response": content,
                }
                return content
            except Exception:
                if attempt >= attempts:
                    raise
                time.sleep(min(2 ** attempt, 5))
        # Unreachable: loop returns or raises
        raise RuntimeError("unreachable")

    def get_last_log(self) -> Optional[Dict[str, Any]]:
        return self._last_log


class TwoStageAnnotator:
    """Two-stage pipeline around an LLM with alignment report context."""

    def __init__(
        self,
        *,
        alignment_report_path: str,
        model: str = "openai:gpt-5-chat-free",
        temperature: float = 0.2,
        stage1_prompt_path: Optional[str] = None,
        stage2_prompt_path: Optional[str] = None,
        max_alignment_chars: int = 12000,
        request_timeout: float = 60.0,
        max_retries: int = 2,
        on_error: str = "skip",  # 'skip' | 'neutral' | 'raise'
    ) -> None:
        self.model = model
        self.temperature = temperature
        raw_report = _read_text_file(alignment_report_path)
        self.alignment_report, self._alignment_truncated = _truncate(raw_report, max_alignment_chars)
        self.stage1_template = Path(stage1_prompt_path or "prompts/stage1_align_explain.md")
        self.stage2_template = Path(stage2_prompt_path or "prompts/stage2_judgement.md")
        self._client = LLMClient()
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.on_error = on_error

    def _load_prompt(self, path: Path, fallback: str) -> str:
        if path.exists():
            return path.read_text(encoding="utf-8")
        return fallback

    def run_on_text(self, text: str, *, language_hint: Optional[str] = None) -> Tuple[Stage1Result, Stage2Result, Dict[str, Any]]:
        """Run Stage 1 and Stage 2 on a single text.

        Returns a tuple of (stage1, stage2, logs) where logs contains raw messages.
        """
        stage1_prompt = self._load_prompt(self.stage1_template, DEFAULT_STAGE1_PROMPT)
        stage2_prompt = self._load_prompt(self.stage2_template, DEFAULT_STAGE2_PROMPT)

        sys_prompt = (
            "You are a careful, culturally sensitive assistant for content safety analysis. "
            "Purpose: identify subculture terms and produce a neutral, non-graphic paraphrase for classification. "
            "Do not give instructions for self-harm, drug use, or eating disorders; avoid explicit details; "
            "Follow the JSON schema strictly."
        )

        # Stage 1
        client = LLMClient()  # use a fresh client per call to be thread-safe
        user_stage1 = _inject(
            stage1_prompt,
            {
                "alignment_report": self.alignment_report,
                "text": text,
                "language": language_hint or "",
            },
        )
        messages1 = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_stage1},
        ]
        s1_raw = client.chat(model=self.model, temperature=self.temperature, messages=messages1, timeout=self.request_timeout, max_retries=self.max_retries)
        s1_obj = _parse_json_from_text(s1_raw) or {}
        s1 = Stage1Result(
            modern_rewrite=str(s1_obj.get("modern_rewrite") or text),
            terms=list(s1_obj.get("terms") or []),
            detected_language=s1_obj.get("detected_language"),
            raw=s1_obj if s1_obj else {"raw": s1_raw},
        )

        # Stage 2 (same conversation)
        user_stage2 = stage2_prompt
        messages2 = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_stage1},
            {"role": "assistant", "content": json.dumps(s1_obj or {"modern_rewrite": s1.modern_rewrite, "terms": s1.terms}, ensure_ascii=False)},
            {"role": "user", "content": user_stage2},
        ]
        s2_raw = client.chat(model=self.model, temperature=self.temperature, messages=messages2, timeout=self.request_timeout, max_retries=self.max_retries)
        s2_obj = _parse_json_from_text(s2_raw) or {}
        labels = s2_obj.get("labels") or {}
        # normalize and coerce to ints with fallbacks
        out_labels = {
            "OD": int(labels.get("OD", 0) or 0),
            "ED": int(labels.get("ED", 0) or 0),
            "SH": int(labels.get("SH", 0) or 0),
        }
        rationale = {
            k: str((s2_obj.get("rationale") or {}).get(k, "")) for k in ("OD", "ED", "SH")
        }
        s2 = Stage2Result(labels=out_labels, rationale=rationale, raw=s2_obj if s2_obj else {"raw": s2_raw})

        logs = {
            "stage1": {
                "messages": messages1,
                "response": s1_raw,
                "truncated_alignment": self._alignment_truncated,
            },
            "stage2": {
                "messages": messages2,
                "response": s2_raw,
            },
        }
        return s1, s2, logs

    def run_dataset(
        self,
        records: List[Dict[str, Any]],
        *,
        text_key: str,
        id_key: Optional[str] = None,
        language_key: Optional[str] = None,
        out_dir: str = "outputs",
        run_prefix: str = "annotations",
        cache_existing_path: Optional[str] = None,
        limit: Optional[int] = None,
        workers: int = 4,
    ) -> str:
        """Run over a list of records and write JSONL outputs with caching."""

        # Prepare run directory and files
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(out_dir) / f"{run_prefix}_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)
        out_path = run_dir / "predictions.jsonl"
        logs_dir = run_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Context fingerprint for resume-dedup
        context_fingerprint = _md5(self.alignment_report + f"|{self.model}|{self.temperature}")

        # Load seen ids from an existing predictions.jsonl if provided
        seen_ids: set = set()
        if cache_existing_path:
            try:
                with open(cache_existing_path, "r", encoding="utf-8") as fin:
                    for line in fin:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            if isinstance(obj, dict) and obj.get("id"):
                                seen_ids.add(str(obj["id"]))
                        except Exception:
                            continue
            except FileNotFoundError:
                pass

        # Build list of records to process (respecting seen_ids and limit)
        to_process: List[Dict[str, Any]] = []
        for rec in records:
            if limit is not None and len(to_process) >= limit:
                break
            txt = str(rec.get(text_key) or "").strip()
            if not txt:
                continue
            rid_local = str(rec.get(id_key)) if id_key and rec.get(id_key) is not None else _md5(txt)
            compound_local = f"{rid_local}:{context_fingerprint}"
            if compound_local in seen_ids:
                continue
            to_process.append(rec)

        total_target = len(to_process)
        pbar = tqdm(total=total_target, desc="Annotating", unit="item") if tqdm is not None else None

        # worker for a single record (runs in thread)
        def _worker(rec: Dict[str, Any]) -> Tuple[str, Any, Any, Any]:
            text_local = str(rec.get(text_key) or "").strip()
            rid_local = str(rec.get(id_key)) if id_key and rec.get(id_key) is not None else _md5(text_local)
            compound_local = f"{rid_local}:{context_fingerprint}"
            lang_local = str(rec.get(language_key)) if language_key else None
            try:
                s1_local, s2_local, raw_logs_local = self.run_on_text(text_local, language_hint=lang_local)
                row_local = {
                    "id": compound_local,
                    "source_id": (rec.get(id_key) if id_key else None),
                    "text": text_local,
                    "stage1": {
                        "modern_rewrite": s1_local.modern_rewrite,
                        "terms": s1_local.terms,
                        "detected_language": s1_local.detected_language,
                    },
                    "stage2": {
                        "labels": s2_local.labels,
                        "rationale": s2_local.rationale,
                    },
                    "_meta": {
                        "model": self.model,
                        "temperature": self.temperature,
                        "alignment_truncated": self._alignment_truncated,
                        "context_fingerprint": context_fingerprint,
                    },
                }
                s1_entry_local = {
                    "id": compound_local,
                    "source_id": rec.get(id_key) if id_key else None,
                    "stage": "stage1",
                    "model": self.model,
                    "temperature": self.temperature,
                    "log": _truncate_deep(raw_logs_local.get("stage1", {}), max_chars=3000),
                }
                s2_entry_local = {
                    "id": compound_local,
                    "source_id": rec.get(id_key) if id_key else None,
                    "stage": "stage2",
                    "model": self.model,
                    "temperature": self.temperature,
                    "log": _truncate_deep(raw_logs_local.get("stage2", {}), max_chars=3000),
                }
                return ("ok", row_local, s1_entry_local, s2_entry_local)
            except Exception as e:
                if self.on_error == "raise":
                    raise
                elif self.on_error == "neutral":
                    # produce neutral placeholder
                    s1_local = Stage1Result(
                        modern_rewrite="[content removed for safety classification]",
                        terms=[],
                        detected_language=None,
                        raw={"error": str(e)},
                    )
                    s2_local = Stage2Result(
                        labels={"OD": 0, "ED": 0, "SH": 0},
                        rationale={"OD": "error", "ED": "error", "SH": "error"},
                        raw={"error": str(e)},
                    )
                    raw_logs_local = {"stage1": {"error": str(e)}, "stage2": {"error": str(e)}}
                    row_local = {
                        "id": compound_local,
                        "source_id": (rec.get(id_key) if id_key else None),
                        "text": text_local,
                        "stage1": {
                            "modern_rewrite": s1_local.modern_rewrite,
                            "terms": s1_local.terms,
                            "detected_language": s1_local.detected_language,
                        },
                        "stage2": {
                            "labels": s2_local.labels,
                            "rationale": s2_local.rationale,
                        },
                        "_meta": {
                            "model": self.model,
                            "temperature": self.temperature,
                            "alignment_truncated": self._alignment_truncated,
                            "context_fingerprint": context_fingerprint,
                        },
                    }
                    s1_entry_local = {
                        "id": compound_local,
                        "source_id": rec.get(id_key) if id_key else None,
                        "stage": "stage1",
                        "model": self.model,
                        "temperature": self.temperature,
                        "log": _truncate_deep(raw_logs_local.get("stage1", {}), max_chars=3000),
                    }
                    s2_entry_local = {
                        "id": compound_local,
                        "source_id": rec.get(id_key) if id_key else None,
                        "stage": "stage2",
                        "model": self.model,
                        "temperature": self.temperature,
                        "log": _truncate_deep(raw_logs_local.get("stage2", {}), max_chars=3000),
                    }
                    return ("ok", row_local, s1_entry_local, s2_entry_local)
                else:
                    # skip
                    return ("err", {"id": compound_local, "source_id": rec.get(id_key) if id_key else None, "error": str(e)}, None, None)

        # Execute workers and write results as they complete
        try:
            with open(out_path, "w", encoding="utf-8") as fout, \
                 open(logs_dir / "stage1.jsonl", "w", encoding="utf-8") as s1f, \
                 open(logs_dir / "stage2.jsonl", "w", encoding="utf-8") as s2f, \
                 open(logs_dir / "errors.jsonl", "w", encoding="utf-8") as erf:
                futures = {}
                with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
                    for rec in to_process:
                        fut = executor.submit(_worker, rec)
                        futures[fut] = rec

                    try:
                        for fut in concurrent.futures.as_completed(futures):
                            res = fut.result()
                            if res[0] == "ok":
                                _, row_local, s1_entry_local, s2_entry_local = res
                                fout.write(json.dumps(row_local, ensure_ascii=False) + "\n")
                                fout.flush()
                                try:
                                    s1f.write(json.dumps(s1_entry_local, ensure_ascii=False) + "\n")
                                    s1f.flush()
                                except Exception:
                                    pass
                                try:
                                    s2f.write(json.dumps(s2_entry_local, ensure_ascii=False) + "\n")
                                    s2f.flush()
                                except Exception:
                                    pass
                            else:
                                errobj = res[1]
                                try:
                                    erf.write(json.dumps(errobj, ensure_ascii=False) + "\n")
                                    erf.flush()
                                except Exception:
                                    pass
                            if pbar is not None:
                                try:
                                    pbar.update(1)
                                except Exception:
                                    pass
                    except KeyboardInterrupt:
                        # cancel pending futures
                        for f in futures:
                            try:
                                f.cancel()
                            except Exception:
                                pass
                        (run_dir / "RUN_INTERRUPTED.txt").write_text(
                            "Annotation interrupted by user (KeyboardInterrupt). Partial predictions saved.",
                            encoding="utf-8",
                        )
                        return str(out_path)
        finally:
            if pbar is not None:
                try:
                    pbar.close()
                except Exception:
                    pass

        return str(out_path)


# -----------------
# Prompt fallbacks
# -----------------

DEFAULT_STAGE1_PROMPT = (
    "You are a Subculture Term Aligner.\n"
    "You are given: (1) an alignment report that lists subculture terms and their meanings,\n"
    "and (2) a user text. Identify any subculture terms/expressions in the text and explain them.\n"
    "Then produce a modern plain-language rewrite of the text that preserves the intended meaning.\n"
    "\n"
    "Return strictly a JSON object with keys: \n"
    "- modern_rewrite: string\n"
    "- terms: array of {term: string, normalized?: string, explanation: string, note?: string}\n"
    "- detected_language?: string\n"
    "Do not add extra keys. Do not include markdown.\n"
    "\n"
    "[Alignment Report]\n"
    "{alignment_report}\n"
    "\n"
    "[User Text]{language_hint}\n"
    "{text}\n"
).replace("{language_hint}", " (language: {language})" )


DEFAULT_STAGE2_PROMPT = (
    "Now, based on the previous JSON (modern_rewrite and terms) in this conversation,\n"
    "assign labels for three independent dimensions using the 0/1/2 ordinal scale:\n"
    "- OD (Overdose / dangerous drug use): 0=Non-concerning, 1=First-person, 2=Third-party.\n"
    "- ED (Eating disorder behaviors): 0=Non-concerning, 1=First-person, 2=Third-party.\n"
    "- SH (Self-harm / suicidal ideation): 0=Non-concerning, 1=First-person, 2=Third-party.\n"
    "\n"
    "Rules:\n"
    "- Use only the content and explanations already present in the conversation.\n"
    "- Be conservative: if ambiguous, choose 0.\n"
    "- First-person means the author is describing their own behavior/thoughts. Third-party means discussing others.\n"
    "\n"
    "Return strictly a JSON object with keys:\n"
    "{\n"
    "  \"labels\": { \"OD\": 0|1|2, \"ED\": 0|1|2, \"SH\": 0|1|2 },\n"
    "  \"rationale\": { \"OD\": string, \"ED\": string, \"SH\": string }\n"
    "}\n"
    "Do not include markdown. Do not add extra keys."
)
