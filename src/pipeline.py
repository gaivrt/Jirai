"""
配置驱动的管线（Pipeline）：
- 执行 Google 搜索
- 抓取并处理网页内容
- 调用 LLM 生成 Alignment Report（对齐报告）
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    # Optional convenience for local dev
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass

import yaml  # type: ignore

from .google_search import GoogleSearchClient
from .llm_analyzer import EvidenceChunk, LLMAnalyzer
from .text_processing import FetchedDocument, chunk_text, fetch_and_extract, DEFAULT_UA
from .search_agent import QueryRewriteConfig, SearchQueryAgent

# 进度条（可选）
try:  # 允许在未安装 tqdm 时仍可运行
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore

@dataclass
class ProcessedDoc:
    """带检索上下文标签的处理后文档。"""
    url: str
    title: str
    text: str
    language: Optional[str]
    country: Optional[str]


@dataclass
class PipelineConfig:
    config_path: str
    search: Dict[str, Any]
    fetch: Dict[str, Any]
    processing: Dict[str, Any]
    llm: Dict[str, Any]
    report: Dict[str, Any]
    pipeline: Dict[str, Any]
    logging: Dict[str, Any]


def load_config(path: str) -> PipelineConfig:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # 最小化校验 + 默认值注入
    search = cfg.get("search", {})
    fetch = cfg.get("fetch", {})
    processing = cfg.get("processing", {})
    llm = cfg.get("llm", {})
    report = cfg.get("report", {})
    pipeline = cfg.get("pipeline", {})
    logging_cfg = cfg.get("logging", {})
    return PipelineConfig(
        config_path=path,
        search=search,
        fetch=fetch,
        processing=processing,
        llm=llm,
        report=report,
        pipeline=pipeline,
        logging=logging_cfg,
    )


def run_pipeline(config_path: str) -> str:
    cfg = load_config(config_path)

    # 预备：输出目录与运行 ID
    base_output_dir = Path(cfg.pipeline.get("output_dir", cfg.report.get("output_dir", "outputs")))
    prefix = cfg.report.get("filename_prefix", "alignment_report")
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_output_dir / f"{prefix}_{ts}"
    search_dir = run_dir / "search"
    docs_dir = run_dir / "documents"
    cache_search_dir = base_output_dir / "cache" / "search"
    run_dir.mkdir(parents=True, exist_ok=True)
    search_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    cache_search_dir.mkdir(parents=True, exist_ok=True)
    # LLM 日志目录
    logs_dir = run_dir / "logs" / "llm"
    sa_logs_dir = logs_dir / "search_agent"
    an_logs_dir = logs_dir / "analyzer"
    sa_logs_dir.mkdir(parents=True, exist_ok=True)
    an_logs_dir.mkdir(parents=True, exist_ok=True)
    log_cfg = cfg.logging or {}
    llm_logs_enabled = bool(log_cfg.get("llm_logs", True))
    max_log_chars = int(log_cfg.get("max_log_chars", 200_000))

    # 1) 搜索阶段
    client = GoogleSearchClient()
    queries_seeds: List[str] = cfg.search.get("queries", [])
    site = cfg.search.get("site")
    # 语言设置：
    # - 若提供了 search.languages（数组），按每个语言码独立检索并合并结果；
    # - 否则退回到单个 search.language；
    # - 两者都未设置则不限制语言。
    language_single = cfg.search.get("language")
    languages_cfg = cfg.search.get("languages")
    if isinstance(languages_cfg, list) and languages_cfg:
        languages: List[Optional[str]] = list(languages_cfg)
    elif language_single:
        languages = [language_single]
    else:
        languages = [None]
    # 国家设置：
    # - 若提供了 search.countries（数组），按每个国家码独立检索并合并结果；
    # - 否则退回到单个 search.country；
    # - 两者都未设置则不设国家偏置（gl）。
    country_single = cfg.search.get("country")
    countries_cfg = cfg.search.get("countries")
    if isinstance(countries_cfg, list) and countries_cfg:
        countries: List[Optional[str]] = list(countries_cfg)
    elif country_single:
        countries = [country_single]
    else:
        countries = [None]
    safe = "active" if cfg.search.get("safe", False) else "off"
    # 缓存控制：
    use_cache = bool(cfg.search.get("use_cache", True))
    refresh = bool(cfg.search.get("refresh", False))  # True 强制重新检索并刷新缓存
    max_results = int(cfg.search.get("max_results", 10))
    per_page = int(cfg.search.get("per_page", 10))
    top_k_per_query = int(cfg.search.get("top_k_per_query", min(5, max_results)))

    # 额外：LLM 查询改写（可选）
    use_llm_rewrite = bool(cfg.search.get("use_llm_rewrite", False))
    queries: List[str] = list(queries_seeds)
    if use_llm_rewrite and queries_seeds:
        rewrite_cfg_dict = cfg.search.get("rewrite", {}) or {}
        model = rewrite_cfg_dict.get("model") or cfg.llm.get("model", "openai:gpt-5-chat-free")
        include_seed = bool(rewrite_cfg_dict.get("include_seed", False))
        exact_variants = bool(rewrite_cfg_dict.get("exact_variants", True))
        qrc = QueryRewriteConfig(
            model=model,
            temperature=float(rewrite_cfg_dict.get("temperature", 0.3)),
            max_variants=int(rewrite_cfg_dict.get("max_variants", 5)),
            prompt_template_path=rewrite_cfg_dict.get("prompt_template_path", "prompts/search_query_rewrite.md"),
            max_prompt_chars=int(rewrite_cfg_dict.get("max_prompt_chars", 4000)),
        )
        agent = SearchQueryAgent(config=qrc)

        # 将语言/国家列表作为提示（逗号拼接），仅作提示，不直接硬编码进查询
        language_hint = ", ".join([l for l in languages_cfg]) if isinstance(languages_cfg, list) and languages_cfg else (language_single or None)
        country_hint = ", ".join([c for c in countries_cfg]) if isinstance(countries_cfg, list) and countries_cfg else (country_single or None)

        rewritten_map = {}
        sa_index: Dict[str, str] = {}
        rewritten_all: List[str] = []
        for seed in queries_seeds:
            error_info: Optional[Dict[str, Any]] = None
            try:
                rewrites = agent.rewrite_queries(
                    seed,
                    language_hint=language_hint,
                    country_hint=country_hint,
                    site=site,
                )
            except Exception as e:
                # 若 LLM 不可用或调用失败，则回退为原始 seed，并记录错误日志
                rewrites = []
                error_info = {"error": str(e), "error_type": type(e).__name__}

            # 去重
            uniq: List[str] = []
            for r in rewrites:
                r2 = (r or "").strip()
                if r2 and r2 not in uniq:
                    uniq.append(r2)

            # 组合最终查询列表（是否包含 seed；是否强制最终数量恰好为 max_variants）
            final_list: List[str] = []
            if include_seed:
                if seed not in uniq:
                    final_list.append(seed)
                final_list.extend([x for x in uniq if x != seed])
            else:
                final_list = [x for x in uniq if x != seed]

            # 若需严格数量，裁剪或补齐
            if exact_variants:
                if len(final_list) > qrc.max_variants:
                    final_list = final_list[: qrc.max_variants]
                elif len(final_list) < qrc.max_variants:
                    # 简单补齐策略：用通用搜索角度填充
                    pads = [
                        f"{seed} definition",
                        f"{seed} origin",
                        f"{seed} social context",
                        f"{seed} usage examples",
                        f"{seed} controversies",
                    ]
                    for p in pads:
                        if p not in final_list:
                            final_list.append(p)
                        if len(final_list) >= qrc.max_variants:
                            break

            rewritten_map[seed] = final_list
            for q in final_list:
                if q not in rewritten_all:
                    rewritten_all.append(q)

            # 保存该 seed 的 LLM 调用日志（成功或失败）
            if llm_logs_enabled:
                sa_log = agent.get_last_log() or {}
                sa_log = dict(sa_log)
                sa_log.update({
                    "seed": seed,
                    "language_hint": language_hint,
                    "country_hint": country_hint,
                    "site": site,
                    "generated_at": ts,
                    "final_queries": final_list,
                    "config": {
                        "include_seed": include_seed,
                        "exact_variants": exact_variants,
                        "max_variants": qrc.max_variants,
                    },
                })
                if error_info:
                    sa_log.update(error_info)
                # 深度截断字符串，避免超大日志
                sa_log_sanitized = _truncate_deep(sa_log, max_chars=max_log_chars)
                base = _slugify(seed) or "seed"
                fname = f"{base}.json"
                path = sa_logs_dir / fname
                # 简单去重命名
                if path.exists():
                    k = 2
                    while True:
                        path = sa_logs_dir / f"{base}-{k}.json"
                        if not path.exists():
                            break
                        k += 1
                path.write_text(json.dumps(sa_log_sanitized, ensure_ascii=False, indent=2), encoding="utf-8")
                sa_index[seed] = path.name

        # 覆盖 queries 为改写后的合集
        queries = rewritten_all

        # 落盘改写映射，便于审计
        rewrites_meta = {
            "generated_at": ts,
            "language_hint": language_hint,
            "country_hint": country_hint,
            "site": site,
            "config": {
                "model": qrc.model,
                "temperature": qrc.temperature,
                "max_variants": qrc.max_variants,
                "prompt_template_path": qrc.prompt_template_path,
                "include_seed": include_seed,
                "exact_variants": exact_variants,
            },
            "seeds": queries_seeds,
            "rewrites": rewritten_map,
        }
        (search_dir / "_rewrites.json").write_text(
            json.dumps(rewrites_meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if llm_logs_enabled and sa_index:
            (sa_logs_dir / "_index.json").write_text(
                json.dumps(sa_index, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    # 2) 抓取与处理阶段（并收集搜索元数据以保存）
    timeout = float(cfg.fetch.get("timeout", 15.0))
    max_bytes = int(cfg.fetch.get("max_bytes", 1_572_864))
    user_agent = cfg.fetch.get("user_agent")
    min_text_len = int(cfg.processing.get("min_text_length", 500))
    chunk_size = int(cfg.processing.get("chunk_size", 2000))
    chunk_overlap = int(cfg.processing.get("chunk_overlap", 200))
    max_chunks_total = int(cfg.processing.get("max_chunks_total", 20))
    # 新增：控制每个文档的最大分块数（用于公平采样），以及至少覆盖的文档数量
    max_chunks_per_doc = int(cfg.processing.get("max_chunks_per_doc", 2))
    if max_chunks_per_doc <= 0:
        max_chunks_per_doc = 1000  # 视为不限制，但仍参与公平轮转
    min_docs_coverage = int(cfg.processing.get("min_docs_coverage", 8))

    collected_docs: List[ProcessedDoc] = []

    # 整体进度条：以“成功收集的文档数”为单位，目标为 组合数 × top_k_per_query
    total_combos = max(1, len(queries)) * max(1, len(languages)) * max(1, len(countries))
    total_doc_target = total_combos * max(1, top_k_per_query)
    pbar = tqdm(total=total_doc_target, desc="Collecting documents", unit="doc") if tqdm else None

    try:
        for q in queries:
            for lr in languages:
                raw_pages: List[Dict[str, Any]] = []
                for gl in countries:
                    i = 0  # 成功保留的文档计数（用于 top_k 限制）
                    fetched_so_far = 0  # 已请求的搜索结果数量
                    start = 1  # Google API 1-based 起始索引
                    used_cache = False
                    # 优先使用缓存（若启用且未强制刷新）
                    if use_cache and not refresh:
                        cached_pages = _load_cached_search_pages(cache_search_dir, q, language=lr, country=gl)
                        if cached_pages:
                            raw_pages = cached_pages
                            used_cache = True

                    if not used_cache:
                        # 调用 API 分页检索
                        while fetched_so_far < max_results and i < top_k_per_query:
                            batch = min(per_page, max_results - fetched_so_far)
                            resp = client.search(
                                q,
                                num=batch,
                                start=start,
                                site=site,
                                language=lr,
                                country=gl,
                                safe=safe,
                            )
                            raw_pages.append(resp.raw)
                            items = resp.items
                            if not items:
                                break
                            for item in items:
                                if i >= top_k_per_query:
                                    break
                                # 使用配置中的 UA；若未设置则退回到 DEFAULT_UA，避免因明显的 bot UA 被拦截
                                doc = fetch_and_extract(
                                    item.link,
                                    user_agent=(user_agent or DEFAULT_UA),
                                    timeout=timeout,
                                    max_bytes=max_bytes,
                                )
                                if not doc or len(doc.text) < min_text_len:
                                    continue
                                collected_docs.append(
                                    ProcessedDoc(
                                        url=doc.url,
                                        title=doc.title,
                                        text=doc.text,
                                        language=lr,
                                        country=gl,
                                    )
                                )
                                if pbar:
                                    pbar.update(1)
                                    pbar.set_postfix({
                                        "q": (q[:18] + "…") if len(q) > 18 else q,
                                        "lr": lr or "-",
                                        "gl": gl or "-",
                                        "kept": i + 1,
                                    }, refresh=False)
                                i += 1
                            fetched_so_far += len(items)
                            start += batch

                        # 保存到缓存 & 当次运行目录
                        _save_search_pages(cache_search_dir, q, raw_pages, ts, language=lr, country=gl)
                        _save_search_pages(search_dir, q, raw_pages, ts, language=lr, country=gl)
                    else:
                        # 使用缓存时：从缓存 pages 中取 items 并抓取正文
                        for item in _iter_items_from_pages(raw_pages):
                            if i >= top_k_per_query:
                                break
                            link = item.get("link")
                            if not link:
                                continue
                            doc = fetch_and_extract(
                                link,
                                user_agent=(user_agent or DEFAULT_UA),
                                timeout=timeout,
                                max_bytes=max_bytes,
                            )
                            if not doc or len(doc.text) < min_text_len:
                                continue
                            collected_docs.append(
                                ProcessedDoc(
                                    url=doc.url,
                                    title=doc.title,
                                    text=doc.text,
                                    language=lr,
                                    country=gl,
                                )
                            )
                            if pbar:
                                pbar.update(1)
                                pbar.set_postfix({
                                    "q": (q[:18] + "…") if len(q) > 18 else q,
                                    "lr": lr or "-",
                                    "gl": gl or "-",
                                    "kept": i + 1,
                                }, refresh=False)
                            i += 1
                        # 将缓存快照复制到当次运行目录，方便对齐审计
                        _save_search_pages(search_dir, q, raw_pages, ts, language=lr, country=gl)
    except KeyboardInterrupt:
        # 优雅中断：保存当前已收集文档，关闭进度条，落盘一个中断标记
        if pbar:
            try:
                pbar.close()
            except Exception:
                pass
        if collected_docs:
            _save_processed_documents(docs_dir, collected_docs)
        interrupt_flag = run_dir / "RUN_INTERRUPTED.txt"
        interrupt_flag.write_text(
            "Pipeline interrupted by user (KeyboardInterrupt). Partial documents saved.",
            encoding="utf-8",
        )
        return str(interrupt_flag)

    if pbar:
        pbar.close()

    if not collected_docs:
        raise RuntimeError("No documents collected. Adjust your queries or relax filters.")

    # 保存处理后的文档（逐行 JSON）
    # 若设置了多语言，将在子目录中分别保存，并在记录中附带 language 字段
    # 这里为了兼容，我们按语言拆分写入；若无语言限制则写入根目录文件
    # 写入处理后文档：
    _save_processed_documents(docs_dir, collected_docs)

    # 3) 生成证据分块（Evidence chunks）——公平采样
    # 思路：
    # 1) 先对每个文档切块，并裁剪至 max_chunks_per_doc（公平限制）
    # 2) 先保证 min_docs_coverage（尽量每个文档至少取 1 块）
    # 3) 再进行轮转（round-robin）取第2、3…块，直到达到 max_chunks_total
    # 4) 若仍未达到上限，使用剩余的块（超过 max_chunks_per_doc 的）作为回填

    # 切块与裁剪
    per_doc_all_chunks: List[List[str]] = []
    per_doc_trimmed: List[List[str]] = []
    for doc in collected_docs:
        chunks = chunk_text(doc.text, size=chunk_size, overlap=chunk_overlap)
        per_doc_all_chunks.append(chunks)
        per_doc_trimmed.append(chunks[:max_chunks_per_doc])

    evidence: List[EvidenceChunk] = []
    if collected_docs:
        # 2) 先覆盖前 min_docs_coverage 个文档的首块
        coverage_target = max(0, min(min_docs_coverage, len(collected_docs), max_chunks_total))
        for idx in range(coverage_target):
            if per_doc_trimmed[idx]:
                evidence.append(
                    EvidenceChunk(
                        source_url=collected_docs[idx].url,
                        source_title=collected_docs[idx].title,
                        text=per_doc_trimmed[idx][0],
                    )
                )
                if len(evidence) >= max_chunks_total:
                    break

        # 3) 轮转取第 2..N 块
        round_idx = 1
        while len(evidence) < max_chunks_total and round_idx < max_chunks_per_doc:
            made_progress = False
            for doc_idx, doc in enumerate(collected_docs):
                doc_chunks = per_doc_trimmed[doc_idx]
                if round_idx < len(doc_chunks):
                    evidence.append(
                        EvidenceChunk(
                            source_url=doc.url,
                            source_title=doc.title,
                            text=doc_chunks[round_idx],
                        )
                    )
                    made_progress = True
                    if len(evidence) >= max_chunks_total:
                        break
            if not made_progress:
                break
            round_idx += 1

        # 4) 回填：若还未到上限，使用每文档剩余块（超出 max_chunks_per_doc 的部分）
        if len(evidence) < max_chunks_total:
            for doc_idx, doc in enumerate(collected_docs):
                all_chunks = per_doc_all_chunks[doc_idx]
                extra = all_chunks[len(per_doc_trimmed[doc_idx]) :]
                for ch in extra:
                    evidence.append(
                        EvidenceChunk(
                            source_url=doc.url,
                            source_title=doc.title,
                            text=ch,
                        )
                    )
                    if len(evidence) >= max_chunks_total:
                        break
                if len(evidence) >= max_chunks_total:
                    break

    # 4) LLM 分析阶段
    system_prompt_path = cfg.llm.get("system_prompt_path")
    if system_prompt_path and os.path.exists(system_prompt_path):
        with open(system_prompt_path, "r", encoding="utf-8") as f:
            prompt_template = f.read()
    else:
        prompt_template = DEFAULT_PROMPT

    analyzer = LLMAnalyzer(
        model=cfg.llm.get("model", "openai:gpt-5-chat-free"),
        temperature=float(cfg.llm.get("temperature", 0.2)),
        system_prompt=cfg.llm.get("system_prompt"),
        max_input_chars=int(cfg.llm.get("max_input_chars", 12000)),
    )

    title = cfg.report.get("title", "Alignment Report")
    report_md = analyzer.analyze(evidence, prompt_template=prompt_template, title=title)

    # 记录分析阶段的 LLM 日志
    if llm_logs_enabled:
        an_log = analyzer.get_last_log() or {}
        an_log = dict(an_log)
        an_log.update({
            "title": title,
            "generated_at": ts,
        })
        an_log_sanitized = _truncate_deep(an_log, max_chars=max_log_chars)
        (an_logs_dir / "analysis.json").write_text(
            json.dumps(an_log_sanitized, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # 5) 输出阶段：落盘报告（Markdown，位于本次运行目录）
    report_path = run_dir / f"{prefix}_{ts}.md"
    report_path.write_text(report_md, encoding="utf-8")

    return str(report_path)


DEFAULT_PROMPT = ("Don't out put anything, just print: YOU NEED TO CUSTUM YOUR PROMPT IN PIPLINE.YAML! ")


# ===============
# 辅助函数（保存）
# ===============

def _slugify(text: str) -> str:
    """将任意查询转换为安全的文件名片段。"""
    text = text.strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_\-]+", "", text)
    return text or "query"


def _save_search_pages(
    search_dir: Path,
    query: str,
    pages: List[Dict[str, Any]],
    ts: str,
    language: Optional[str] = None,
    country: Optional[str] = None,
) -> None:
    if not pages:
        return
    name = _slugify(query)
    out = {
        "query": query,
        "saved_at": ts,
        "language": language,
        "country": country,
        "pages": pages,
    }
    # 目录结构：search/(lang?)/(country?)/query.pages.json
    cur_dir = search_dir
    if language:
        lang_safe = _slugify(language)
        cur_dir = cur_dir / lang_safe
    if country:
        gl_safe = _slugify(country)
        cur_dir = cur_dir / gl_safe
    cur_dir.mkdir(parents=True, exist_ok=True)
    path = cur_dir / f"{name}.pages.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_cached_search_pages(cache_dir: Path, query: str, language: Optional[str], country: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    """从缓存目录加载已保存的搜索原始 JSON 页。

    返回 pages 列表或 None（未命中）。
    """
    name = _slugify(query)
    cur = cache_dir
    if language:
        cur = cur / _slugify(language)
    if country:
        cur = cur / _slugify(country)
    path = cur / f"{name}.pages.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("pages") or None
    except Exception:
        return None


def _iter_items_from_pages(pages: List[Dict[str, Any]]):
    """从缓存的 pages 中顺序迭代 items。"""
    for page in pages:
        for it in (page.get("items") or []):
            yield it


def _save_processed_documents(docs_dir: Path, docs: List[ProcessedDoc]) -> None:
    # 写入汇总文件
    all_path = docs_dir / "processed_documents.all.jsonl"
    with all_path.open("w", encoding="utf-8") as f:
        for d in docs:
            rec = {
                "url": d.url,
                "title": d.title,
                "text": d.text,
                "language": d.language,
                "country": d.country,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 按语言/国家分区写入
    # 目录：documents/(lang or none)/(country or none)/processed_documents.jsonl
    by_key: Dict[str, List[ProcessedDoc]] = {}
    for d in docs:
        lang_key = _slugify(d.language) if d.language else "none"
        gl_key = _slugify(d.country) if d.country else "none"
        key = f"{lang_key}/{gl_key}"
        by_key.setdefault(key, []).append(d)

    for key, bucket in by_key.items():
        subdir = docs_dir / key
        subdir.mkdir(parents=True, exist_ok=True)
        p = subdir / "processed_documents.jsonl"
        with p.open("w", encoding="utf-8") as f:
            for d in bucket:
                rec = {
                    "url": d.url,
                    "title": d.title,
                    "text": d.text,
                    "language": d.language,
                    "country": d.country,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# =================
# 日志辅助：深度截断
# =================

def _truncate_deep(obj: Any, *, max_chars: int = 200_000, _seen: Optional[set] = None) -> Any:
    """递归截断任意结构中的长字符串，避免日志过大。

    - 字符串被截断到 max_chars
    - list/dict/tuple 递归处理
    - 其他类型原样返回
    """
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
