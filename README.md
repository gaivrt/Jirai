# Purpose | 目的

This repository is the first part of codes for the structure.

## Google search (Programmable Search API)

This project now includes a small, typed client for Google Programmable Search (Custom Search JSON API) and a CLI to query it.

中文文档请见：`README_zh.md`。

### Setup

1) Install dependencies

```
pip install -r requirements.txt
```

2) Configure credentials

Provide these environment variables (e.g., in a local `.env` file):

- `GOOGLE_API_KEY`: Your Google API key
- `GOOGLE_CSE_ID` (or `GOOGLE_CX`): Your Programmable Search Engine ID

You can create an API key and a Programmable Search Engine (PSE) at:
https://developers.google.com/custom-search/v1/overview

Example `.env` entries:

```
GOOGLE_API_KEY=your_api_key_here
GOOGLE_CSE_ID=your_cse_id_here
```

### Usage (config-driven pipeline)

Edit `config/pipeline.yaml` to define your queries and options, then run:

```
python -m src.main
```

This will:
- query Google Programmable Search per your config
- fetch and extract text from the top results
- analyze the evidence with your configured LLM
- write an Alignment Report to `outputs/`

To use a different config file:

```
python -m src.main --config config/my_pipeline.yaml
```

### Code

- `src/google_search.py`: Typed Google Search client with retries and helpful errors
- `src/text_processing.py`: Fetch and extract HTML text, chunk utilities
- `src/llm_analyzer.py`: LLM-based Alignment Report generator (uses `aisuite`)
- `src/pipeline.py`: Orchestrates search -> fetch -> analyze -> report via YAML config
- `src/main.py`: Entry point. Default runs the pipeline using `config/pipeline.yaml`. Includes a debug `google-search` subcommand.
- `src/annotator.py`: Two-stage alignment-assisted annotator (Part B). Stage 1 aligns/explains; Stage 2 judges OD/ED/SH.
- `src/evaluate.py`: Dataset loading from .xlsx and Macro F1 computation.
- `config/pipeline.yaml`: Example pipeline configuration
- `prompts/alignment_report_prompt.md`: Prompt template used by the analyzer
 - `prompts/stage1_align_explain.md`: Stage 1 prompt template (identify/explicate terms, modern rewrite)
 - `prompts/stage2_judgement.md`: Stage 2 prompt template (OD/ED/SH labels)

### Part B: Two-stage annotation and evaluation

Given an existing Alignment Report (Markdown) and a dataset in Excel (`data/OD_Multilingual/*.xlsx`), you can run the two-stage annotator and optionally compute Macro F1 if gold labels are present.

Run the annotator:

```
python -m src.main annotate \
	outputs/alignment_report_YYYYMMDD_HHMMSS/alignment_report_YYYYMMDD_HHMMSS.md \
	data/OD_Multilingual/od_data_ori_J.xlsx \
	--model openai:gpt-5-chat-free \
	--workers 4 \
	--eval
```

Notes:
- The tool auto-detects text and label columns; you can override with `--text-col/--od-col/--ed-col/--sh-col`.
- Outputs will be written under `outputs/annotations_*/predictions.jsonl` with per-item logs.
- To evaluate Macro F1, pass `--eval` and ensure the dataset contains OD/ED/SH columns with values 0/1/2.

Merged evaluation options:
- To evaluate on the combination of a previous run and the new run when resuming, add `--eval --eval-merged --resume-from <prev_predictions.jsonl>`.
- Or evaluate any set of prediction files with the standalone subcommand:

```
python -m src.main evaluate \
	data/OD_Multilingual/od_data_ori_J.xlsx \
	outputs/annotations_20251028_000348/predictions.jsonl \
	outputs/annotations_20251028_010101/predictions.jsonl \
	--dedupe-by source_id --keep last --id-key _row
```

Performance & controls:
- `--workers N`: run parallel LLM calls with N worker threads (default 4). Start small to avoid provider rate limits.
- `--limit K`: process only the first K rows (good for smoke tests).
- `--resume-from <predictions.jsonl>`: skip records already annotated in a previous run (id + context fingerprint).
- `--timeout` and `--retries`: per-request timeout and retry count.
- `--on-error {skip|neutral|raise}`: choose how to handle provider blocks or errors per item.
