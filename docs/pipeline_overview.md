# Jirai: Research-Oriented Web Evidence → Alignment Report

This document explains the big-picture logic of our research pipeline. It focuses on the experimental flow rather than per-file internals. Think of it as the Part A in `/docs/basic_information.md`:

Input Field → Websearch Agent → Research Results → Report Agent → Alignment Report

Below we walk through each step, show what goes in and comes out, and reference real artifacts produced under `outputs/` and `logs/`.

---

## What this program does (in one paragraph)
You give a topic as short keywords (seeds). The Websearch Agent (LLM) expands them into several precise search queries. We call Google Programmable Search for each query (across languages/countries if configured), fetch the pages, extract and de-noise readable text, then build a balanced evidence pack. Finally, the Report Agent (LLM) writes an Alignment Report grounded only in that evidence. Every step saves its artifacts for audit: raw search JSON, processed documents, LLM prompts and outputs, and the final report.

---

## 1) Input Field
- Source: `config/pipeline.yaml`
- Key inputs:
  - `search.queries` (seeds you type)
  - `search.languages`, `search.countries` (optional filters)
  - `search.use_llm_rewrite` and `search.rewrite.*` (rewrite scale/behavior)

Example (active config):
```yaml
search:
  queries:
    - "Glossary of Terms Related to 'Jirai Onna'"
  languages: ["lang_en"]
  countries: ["cn", "jp", "us"]
  use_llm_rewrite: true
  rewrite:
    max_variants: 3
    include_seed: false
    exact_variants: true
```

What this means: we start with ONE seed, but we want EXACTLY 3 variants per seed, and we don’t count the original seed among the 3.

---

## 2) Websearch Agent (LLM-based query rewrite)
- Purpose: Expand a human seed into multiple, high-signal queries exploring different angles (definition, origin, context, etc.).
- Input:
  - seed string
  - hints: `languages`, `countries`, optional `site`
  - config: `max_variants`, `include_seed`, `exact_variants`
- Output:
  - `final_queries`: the exact set of queries we’ll actually send to Google.
  - Full LLM request/response logs are saved.

Where to see the logs (real example):
- Folder: `outputs/alignment_report_20251024_075511/logs/llm/search_agent/`
- Index: `_index.json` → maps seed → per-seed log file
- Per-seed log: `glossary_of_terms_related_to_jirai_onna.json`

Excerpt (shortened):
```json
{
  "seed": "Glossary of Terms Related to 'Jirai Onna'",
  "language_hint": "lang_en",
  "country_hint": "cn, jp, us",
  "final_queries": [
    "Glossary of Terms Related to 'Jirai Onna' definition",
    "Glossary of Terms Related to 'Jirai Onna' origin",
    "Glossary of Terms Related to 'Jirai Onna' social context"
  ],
  "config": { "include_seed": false, "exact_variants": true, "max_variants": 3 },
  "error": "...",
  "error_type": "KeyError"
}
```
Notes:
- Even if the model returns fewer items occasionally, we enforce exact `max_variants` via fallback fillers (so downstream search scale is predictable).
- Any LLM error is recorded here, and we still produce `final_queries` per policy.

---

## 3) Research Results (Google search → fetch → extract)
- Purpose: For each query × language × country, run Google search, snapshot raw JSON, fetch top-k pages, and extract clean text.
- Inputs:
  - The `final_queries` list from Websearch Agent
  - `search.languages` (e.g., `lang_en`) and `search.countries` (e.g., `us`)
- Outputs:
  - Raw search pages JSON
  - Processed documents (JSONL): url, title, cleaned text

Where to see them:
- Raw search snapshots per (lang/country/query):
  - `outputs/alignment_report_20251024_075511/search/lang_en/us/glossary_of_terms_related_to_jirai_onna_origin.pages.json`
- Processed documents per (lang/country):
  - `outputs/alignment_report_20251024_075511/documents/lang_en/us/processed_documents.jsonl`

Search snapshot excerpt:
```json
{
  "query": "Glossary of Terms Related to 'Jirai Onna' origin",
  "language": "lang_en",
  "country": "us",
  "pages": [
    {
      "queries": { "request": [{ "totalResults": "969", "gl": "cn", "lr": "lang_en" }] },
      "items": [
        { "title": "Can someone explain what a \"Mine Woman\" (地雷女) is? : r/japan",
          "link": "https://www.reddit.com/r/japan/comments/g6dj45/..." }
      ]
    }
  ]
}
```

Processed document excerpt (first lines):
```jsonl
{"url": "https://www.reddit.com/r/japan/comments/g6dj45/...", "title": "Reddit - The heart of the internet", "text": "Reddit - The heart of the internet...", "language": "lang_en", "country": "us"}
{"url": "https://j-fashion.fandom.com/wiki/Jirai_Kei", "title": "Jirai Kei | Japanese Fashion Wikia | Fandom", "text": "Jirai Kei | Japanese Fashion Wikia | Fandom...", "language": "lang_en", "country": "us"}
```

Implementation notes (high level):
- We stream HTTP bodies with size caps (default ~1.5 MiB) and explicit connect/read timeouts to avoid stalls on slow/huge pages.
- Text is cleaned and chunked; we then sample chunks fairly across many sources for evidence diversity.

---

## 4) Report Agent (LLM analysis)
- Purpose: Turn sampled evidence into a structured Alignment Report grounded only in the provided excerpts/URLs.
- Input to LLM:
  - A system message with analysis role
  - A user message that embeds the prompt template and an Evidence section containing: title, source URL(s), and excerpt chunks
- Output:
  - Markdown report
  - Full LLM request/response logs (including evidence truncation stats)

Where to see logs and output:
- LLM logs: `outputs/alignment_report_20251024_075511/logs/llm/analyzer/analysis.json`
  - Contains `request.messages[0]` (system) and `request.messages[1]` (user with embedded Evidence), plus `response.content`.
- Final report: `outputs/alignment_report_20251024_075511/alignment_report_20251024_075511.md`

System message excerpt (from logs):
```json
{"role": "system", "content": "You are a careful, objective analyst..."}
```

Report file excerpt (header):
```markdown
**Report on the "Jirai Kei" culture**

**Introduction:**
Jirai Kei (地雷系), which translates to "landmine type," is a Japanese youth subculture...
```

The report also includes a Sources section listing all URLs it used, for traceability.

---

## 5) Alignment Report (artifact + audit trail)
- The final artifact is a Markdown file in the run directory (e.g., `alignment_report_20251024_075511.md`).
- Each run also saves:
  - `search/_rewrites.json` (seed → final queries, config summary)
  - `search/lang_*/country/*/*.pages.json` (raw search snapshots)
  - `documents/**/*.jsonl` (cleaned text by language/country)
  - `logs/llm/search_agent/*.json` (per-seed rewrite I/O)
  - `logs/llm/analyzer/analysis.json` (report generation I/O)

This lets you audit the whole chain: from the seed you typed, through LLM rewrites, to which pages were searched/fetched, what text was extracted, exactly what the report LLM saw, and the final Markdown it produced.

---

## Scale knobs you likely care about
- Rewrite breadth per seed: `search.rewrite.max_variants` (+ `include_seed`, `exact_variants`)
- Search breadth per combo: `search.top_k_per_query`, `search.max_results`
- Evidence balance: `processing.max_chunks_total`, `max_chunks_per_doc`, `min_docs_coverage`
- Input size cap to LLM: `llm.max_input_chars`
- Network robustness: `fetch.timeout`, `fetch.max_bytes`, `fetch.user_agent`
- Caching: `search.use_cache`, `search.refresh`

---

## One concrete end-to-end example (paths from 2025-10-24 run)
- Seed: `Glossary of Terms Related to 'Jirai Onna'`
- Rewrites used:
  - `final_queries` = `definition`, `origin`, `social context` variants (see `_rewrites.json`)
- One search snapshot:
  - `search/lang_en/us/glossary_of_terms_related_to_jirai_onna_origin.pages.json` (969 totalResults; first link is a Reddit thread explaining 地雷女)
- Processed docs (subset):
  - Reddit explainer, Fandom wikis (Aesthetics / J‑Fashion), Substack essays, Wiktionary entries, etc. (see `documents/lang_en/us/processed_documents.jsonl`)
- Report: `alignment_report_20251024_075511.md` (confidence: 0.9)

---

## FAQ
- Q: Do we pass URLs explicitly to the LLM?
  - A: Yes—inside the Evidence block of the user message. Each chunk carries its source URL and title, then its text excerpt.
- Q: If the rewrite LLM fails?
  - A: We log the error and synthesize `final_queries` using safe fallbacks so downstream still runs. See `logs/llm/search_agent/*.json` for details.
- Q: How do I reduce cost/time?
  - A: Lower `search.rewrite.max_variants`, narrow languages/countries, reduce `top_k_per_query`, and lower evidence caps. You can also enable `search.use_cache` for repeated runs.
