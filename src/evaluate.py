"""
Evaluation utilities for OD/ED/SH annotations.

Provides:
- load_dataset_from_excel: load rows from .xlsx with flexible column detection
- compute_macro_f1: compute Macro F1 for each dimension and overall average
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class DatasetSpec:
    text_col: str
    od_col: Optional[str]
    ed_col: Optional[str]
    sh_col: Optional[str]


def _try_candidates(cols: List[str], candidates: List[str]) -> Optional[str]:
    low = {c.lower(): c for c in cols}
    for name in candidates:
        k = name.lower()
        if k in low:
            return low[k]
    return None


def load_dataset_from_excel(path: str, *, text_col: Optional[str] = None, od_col: Optional[str] = None, ed_col: Optional[str] = None, sh_col: Optional[str] = None) -> Tuple[List[Dict[str, Any]], DatasetSpec]:
    """Load .xlsx into a list of dicts; attempt to auto-detect columns if not specified.

    Requires pandas and openpyxl installed in the environment.
    """
    import pandas as pd  # type: ignore

    df = pd.read_excel(path)
    cols: List[str] = list(df.columns)

    # Auto-detect text column
    text_candidates = [
        text_col,
        "text", "content", "post", "原文", "文本", "内容", "tweet", "message",
    ]
    text_candidates = [c for c in text_candidates if c]
    tc = _try_candidates(cols, text_candidates)
    if not tc:
        raise ValueError(f"Cannot auto-detect text column from candidates: {text_candidates}. Available: {cols}")

    # Auto-detect label columns (optional)
    od_candidates = [od_col, "OD", "od", "药物过量"]
    ed_candidates = [ed_col, "ED", "ed", "饮食失调"]
    sh_candidates = [sh_col, "SH", "sh", "自我伤害"]
    oc = _try_candidates(cols, [c for c in od_candidates if c])
    ec = _try_candidates(cols, [c for c in ed_candidates if c])
    sc = _try_candidates(cols, [c for c in sh_candidates if c])

    spec = DatasetSpec(text_col=tc, od_col=oc, ed_col=ec, sh_col=sc)

    # Build list of records
    records: List[Dict[str, Any]] = []
    py_rows: List[Dict[str, Any]] = df.to_dict(orient="records")  # type: ignore[no-redef]
    import pandas as pd  # type: ignore

    def _to_int_or_none(v: Any) -> Optional[int]:
        try:
            if v is None:
                return None
            if isinstance(v, float) and pd.isna(v):
                return None
            if isinstance(v, str) and v.strip() == "":
                return None
            return int(float(v))
        except Exception:
            return None
    for i, row in enumerate(py_rows):
        rec: Dict[str, Any] = {"_row": int(i), "text": str(row.get(tc) or "")}
        if oc and oc in row:
            rec["OD"] = _to_int_or_none(row.get(oc))
        if ec and ec in row:
            rec["ED"] = _to_int_or_none(row.get(ec))
        if sc and sc in row:
            rec["SH"] = _to_int_or_none(row.get(sc))
        records.append(rec)

    return records, spec


def compute_macro_f1(
    gold: List[Dict[str, Any]],
    pred: List[Dict[str, Any]],
    *,
    id_key: str = "_row",
) -> Dict[str, Any]:
    """Compute Macro F1 per dimension and overall.

    Expects gold and pred lists to contain entries with a common id_key linking them.
    Each pred entry should have stage2.labels dict.
    """
    import numpy as np  # type: ignore
    from sklearn.metrics import f1_score  # type: ignore

    # Index predictions by id, with fallbacks: prefer id_key, then 'source_id'
    pred_map: Dict[str, Dict[str, Any]] = {}
    for p in pred:
        pid = p.get(id_key)
        if pid is None:
            pid = p.get("source_id")
        if pid is None:
            continue
        pred_map[str(pid)] = p

    dims = ["OD", "ED", "SH"]
    f1s: Dict[str, float] = {}
    counts: Dict[str, int] = {d: 0 for d in dims}

    for dim in dims:
        y_true: List[int] = []
        y_pred: List[int] = []
        for g in gold:
            gid = g.get(id_key)
            if str(gid) not in pred_map:
                continue
            if g.get(dim) is None:
                continue
            y_true.append(int(g[dim]))
            plabels = pred_map[str(gid)].get("stage2", {}).get("labels", {})
            y_pred.append(int(plabels.get(dim, 0)))
        if y_true:
            f1s[dim] = float(f1_score(y_true, y_pred, average="macro"))
            counts[dim] = len(y_true)
        else:
            f1s[dim] = float("nan")

    vals = [v for v in f1s.values() if v == v]  # drop NaNs
    overall = float(sum(vals) / len(vals)) if vals else float("nan")
    return {
        "macro_f1": f1s,
        "overall_macro_f1": overall,
        "counts": counts,
    }


def load_predictions_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def merge_predictions_jsonl(
    paths: List[str],
    *,
    dedupe_key: str = "source_id",
    keep: str = "last",  # 'last' or 'first'
) -> List[Dict[str, Any]]:
    """Merge multiple predictions.jsonl files into a single list with de-duplication.

    - dedupe_key: which key to use for identity ('source_id' recommended). If missing, falls back to 'id'.
    - keep: if 'last', later files override earlier ones; if 'first', keep the earliest occurrence.
    """
    assert keep in ("last", "first"), "keep must be 'last' or 'first'"
    merged: Dict[str, Dict[str, Any]] = {}

    def _key_of(p: Dict[str, Any]) -> Optional[str]:
        if dedupe_key in p and p[dedupe_key] is not None:
            return str(p[dedupe_key])
        if "id" in p and p["id"] is not None:
            return str(p["id"])
        return None

    for path in paths:
        for p in load_predictions_jsonl(path):
            k = _key_of(p)
            if k is None:
                continue
            if keep == "first" and k in merged:
                continue
            merged[k] = p
    return list(merged.values())
