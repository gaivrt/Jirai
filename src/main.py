"""
Jirai 主入口。

两种模式：
1) 配置驱动的管线（默认，无子命令）：
    - 读取 YAML 配置，依次执行 Google 搜索、抓取与文本处理、LLM 分析，输出对齐报告。

2) google-search 子命令（调试用）：
    - 直接调用 Google 自定义搜索 JSON API 进行检索。

常见环境变量：
- GOOGLE_API_KEY
- GOOGLE_CSE_ID（或 GOOGLE_CX）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass

from .google_search import GoogleSearchClient, GoogleSearchConfigError, GoogleSearchError
from .pipeline import run_pipeline
from .annotator import TwoStageAnnotator
from .evaluate import load_dataset_from_excel, compute_macro_f1, load_predictions_jsonl, merge_predictions_jsonl


def _cmd_google_search(args: argparse.Namespace) -> int:
    try:
        client = GoogleSearchClient()
    except GoogleSearchConfigError as e:
        print(
            f"Configuration error: {e}\n\n"
            "Set GOOGLE_API_KEY and GOOGLE_CSE_ID (or GOOGLE_CX) in your environment or .env file.",
            file=sys.stderr,
        )
        return 2

    try:
        if args.max_results and args.max_results > args.num:
            # Multi-page iteration
            results = []
            for item in client.iter_results(
                args.query,
                max_results=args.max_results,
                site=args.site,
                exact_terms=args.exact,
                exclude_terms=args.exclude,
                language=args.language,
                country=args.country,
                safe="active" if args.safe else "off",
                page_size=args.num,
            ):
                results.append(item.__dict__)

            if args.json:
                print(json.dumps({"items": results}, ensure_ascii=False, indent=2))
            else:
                for i, it in enumerate(results, start=1):
                    print(f"{i}. {it.get('title')}")
                    print(f"   {it.get('link')}")
                    snippet = it.get("snippet")
                    if snippet:
                        print(f"   {snippet}")
                    print()
        else:
            # 单页检索
            resp = client.search(
                args.query,
                num=args.num,
                start=args.start,
                site=args.site,
                exact_terms=args.exact,
                exclude_terms=args.exclude,
                language=args.language,
                country=args.country,
                safe="active" if args.safe else "off",
            )
            if args.json:
                # Emit a concise structured response
                payload: Dict[str, Any] = {
                    "total_results": resp.total_results,
                    "search_time": resp.search_time,
                    "items": [it.__dict__ for it in resp.items],
                }
                print(json.dumps(payload, ensure_ascii=False, indent=2))
            else:
                header = (
                    f"Results: {len(resp.items)} / ~{resp.total_results} in {resp.search_time:.2f}s\n"
                )
                print(header)
                for i, it in enumerate(resp.items, start=1):
                    print(f"{i}. {it.title}")
                    print(f"   {it.link}")
                    if it.snippet:
                        print(f"   {it.snippet}")
                    print()
    except GoogleSearchError as e:
        print(f"Search error: {e}", file=sys.stderr)
        return 1

    return 0


def _cmd_annotate(args: argparse.Namespace) -> int:
    # Load dataset
    records, spec = load_dataset_from_excel(
        args.data,
        text_col=args.text_col,
        od_col=args.od_col,
        ed_col=args.ed_col,
        sh_col=args.sh_col,
    )
    if not records:
        print("No records found in dataset.", file=sys.stderr)
        return 2

    limit = args.limit if getattr(args, "limit", 0) else None
    annot = TwoStageAnnotator(
        alignment_report_path=args.alignment_report,
        model=args.model,
        temperature=float(args.temperature),
        request_timeout=float(getattr(args, "timeout", 60.0)),
        max_retries=int(getattr(args, "retries", 2)),
        on_error=str(getattr(args, "on_error", "skip")),
    )
    out_path = annot.run_dataset(
        records,
        text_key="text",
        id_key="_row",
        language_key=None,
        out_dir=args.output_dir,
        run_prefix="annotations",
        cache_existing_path=getattr(args, "resume_from", None),
        limit=limit,
        workers=getattr(args, "workers", 4),
    )
    print(f"Predictions written to: {out_path}")

    # Optional evaluation
    if args.do_eval and spec.od_col and spec.ed_col and spec.sh_col:
        # If requested, merge resume-from predictions with the new run for evaluation
        if getattr(args, "eval_merged", False) and getattr(args, "resume_from", None):
            preds = merge_predictions_jsonl([args.resume_from, str(out_path)], dedupe_key="source_id", keep="last")
        else:
            preds = load_predictions_jsonl(out_path)
        metrics = compute_macro_f1(records, preds, id_key="_row")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    elif args.do_eval:
        print("Evaluation requested but label columns were not found. Skipping eval.")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    # Load dataset
    records, spec = load_dataset_from_excel(args.data)
    if not records:
        print("No records found in dataset.", file=sys.stderr)
        return 2

    # Merge predictions
    pred_list = merge_predictions_jsonl(args.predictions, dedupe_key=args.dedupe_by, keep=args.keep)
    # Compute metrics
    metrics = compute_macro_f1(records, pred_list, id_key=args.id_key)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jirai",
        description="Jirai CLI: config-driven pipeline or utilities",
    )
    parser.add_argument(
        "--config",
        "-c",
        default="config/pipeline.yaml",
        help="Path to pipeline config file (YAML)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_search = sub.add_parser(
        "google-search",
        help="Search the web via Google Programmable Search",
        description="Search the web via Google Programmable Search (Custom Search JSON API)",
    )
    p_search.add_argument("query", help="Search query string")
    p_search.add_argument("--num", type=int, default=10, help="Results per page (1-10)")
    p_search.add_argument("--start", type=int, default=1, help="Start index (1-based)")
    p_search.add_argument("--max-results", type=int, default=0, help="Fetch up to N results across pages")
    p_search.add_argument("--site", type=str, default=None, help="Restrict to a site/domain, e.g. example.com")
    p_search.add_argument("--exact", type=str, default=None, help="Exact phrase to include")
    p_search.add_argument("--exclude", type=str, default=None, help="Terms to exclude")
    p_search.add_argument("--language", type=str, default=None, help='Language restrict, e.g. "lang_en"')
    p_search.add_argument("--country", type=str, default=None, help='Country code for geolocation, e.g. "us"')
    p_search.add_argument("--safe", action="store_true", help="Enable safe search")
    p_search.add_argument("--json", action="store_true", help="Output JSON instead of text")
    p_search.set_defaults(func=_cmd_google_search)

    # Part B: two-stage annotation/evaluation
    p_annot = sub.add_parser(
        "annotate",
        help="Run two-stage alignment-assisted annotation on a dataset",
    )
    p_annot.add_argument("alignment_report", help="Path to the alignment report markdown file")
    p_annot.add_argument("data", help="Path to dataset .xlsx file (OD_Multilingual)")
    p_annot.add_argument("--text-col", dest="text_col", default=None, help="Text column name (auto-detect if omitted)")
    p_annot.add_argument("--od-col", dest="od_col", default=None, help="OD label column name (optional)")
    p_annot.add_argument("--ed-col", dest="ed_col", default=None, help="ED label column name (optional)")
    p_annot.add_argument("--sh-col", dest="sh_col", default=None, help="SH label column name (optional)")
    p_annot.add_argument("--model", default="openai:gpt-5-chat-free", help="aisuite model id")
    p_annot.add_argument("--temperature", type=float, default=0.2)
    p_annot.add_argument("--output-dir", dest="output_dir", default="outputs", help="Base output directory")
    p_annot.add_argument("--limit", type=int, default=0, help="Limit number of rows (0 = all)")
    p_annot.add_argument("--eval", dest="do_eval", action="store_true", help="Compute Macro F1 if gold labels found")
    p_annot.add_argument("--timeout", type=float, default=60.0, help="Per-request timeout (seconds)")
    p_annot.add_argument("--retries", type=int, default=2, help="Number of retries on request failure")
    p_annot.add_argument("--resume-from", dest="resume_from", default=None, help="Path to existing predictions.jsonl to skip processed items")
    p_annot.add_argument("--on-error", dest="on_error", choices=["skip", "neutral", "raise"], default="skip", help="Policy when a request fails or is blocked")
    p_annot.add_argument("--workers", type=int, default=4, help="Number of worker threads for parallel annotation (default: 4)")
    p_annot.add_argument("--eval-merged", action="store_true", help="If set with --eval and --resume-from, compute Macro F1 on merged predictions (resume + new), last-wins by source_id")
    p_annot.set_defaults(func=_cmd_annotate)

    # Standalone evaluation subcommand
    p_eval = sub.add_parser(
        "evaluate",
        help="Compute Macro F1 given a dataset (.xlsx) and one or more predictions.jsonl files",
    )
    p_eval.add_argument("data", help="Path to dataset .xlsx file (OD_Multilingual)")
    p_eval.add_argument("predictions", nargs="+", help="One or more predictions.jsonl files (later override earlier if deduping)")
    p_eval.add_argument("--dedupe-by", dest="dedupe_by", choices=["source_id", "id"], default="source_id", help="Key to identify samples in predictions for merging (default: source_id)")
    p_eval.add_argument("--keep", choices=["last", "first"], default="last", help="When duplicates exist, keep the last or first occurrence (default: last)")
    p_eval.add_argument("--id-key", dest="id_key", default="_row", help="Dataset id key used to match predictions (default: _row)")
    p_eval.set_defaults(func=_cmd_evaluate)

    return parser


def main(argv: Any = None) -> int:
    # If no args provided, run pipeline by default (config-driven)
    if argv is None and len(sys.argv) == 1:
        out_path = run_pipeline("config/pipeline.yaml")
        print(f"Report written to: {out_path}")
        return 0

    parser = build_parser()
    # Allow running pipeline without specifying subcommand by passing --config only
    # e.g., python -m src.main --config config/my.yaml
    if argv is None and len(sys.argv) == 3 and sys.argv[1] in ("--config", "-c"):
        args = parser.parse_args(argv)
    else:
        args = parser.parse_args(argv)

    if getattr(args, "command", None) is None:
        out_path = run_pipeline(args.config)
        print(f"Report written to: {out_path}")
        return 0

    return args.func(args)  # type: ignore[attr-defined]


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
