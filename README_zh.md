# 目的

本仓库提供一个“配置驱动”的信息检索与对齐报告（Alignment Report）生成流程示例：

1. 使用 Google 可编程搜索（Custom Search JSON API）进行检索
2. 抓取网页并提取正文文本，进行基础清洗与分块
3. 通过 LLM（示例中使用 aisuite 客户端）对证据进行分析，输出 Markdown 报告

> 英文版请参见 `README.md`。

---

## 目录结构概览

- `src/google_search.py`：Google 搜索 API 客户端
- `src/text_processing.py`：网页抓取、正文提取与分块
- `src/llm_analyzer.py`：基于 aisuite 的 LLM 分析器，生成报告
- `src/pipeline.py`：管线编排（读取配置 → 搜索 → 抓取处理 → 分析 → 写报告）
- `src/main.py`：入口程序。默认直接运行管线；包含调试用 `google-search` 子命令
- `src/annotator.py`：两阶段的对齐辅助标注（Part B）。阶段1对齐/解释，阶段2判定 OD/ED/SH。
- `src/evaluate.py`：从 .xlsx 载入数据并计算 Macro F1。
- `config/pipeline.yaml`：示例管线配置文件，你可以按需修改
- `prompts/alignment_report_prompt.md`：报告提示词模板（可替换）
 - `prompts/stage1_align_explain.md`：阶段1提示词（识别术语并现代改写）
 - `prompts/stage2_judgement.md`：阶段2提示词（OD/ED/SH 三维度打标）

---

## 安装依赖

建议使用你已有的 Conda 环境（例如 `Jirai_env`）。在仓库根目录执行：

```bash
python -m pip install -r requirements.txt
```

依赖说明：
- `requests`：HTTP 客户端
- `python-dotenv`：自动加载 `.env`
- `PyYAML`：解析 YAML 配置
- `beautifulsoup4`：解析 HTML，提取正文

---

## 配置凭据

使用 Google 可编程搜索需要：
- `GOOGLE_API_KEY`：Google API Key
- `GOOGLE_CSE_ID`（或 `GOOGLE_CX`）：可编程搜索引擎 ID（cx）

你可以在 `.env` 中设置（推荐本地开发）：

```env
GOOGLE_API_KEY=your_api_key_here
GOOGLE_CSE_ID=your_cse_id_here
```

也可在 Shell 里导出：

```bash
export GOOGLE_API_KEY="your_api_key_here"
export GOOGLE_CSE_ID="your_cse_id_here"
```

另外，LLM 调用所需的密钥（如 `OPENAI_API_KEY` 等）也需按 aisuite 的要求配置在环境中。

> 创建 API key 与搜索引擎（PSE）请参考官方文档：
> https://developers.google.com/custom-search/v1/overview

---

## 配置驱动的运行方式（推荐）

编辑 `config/pipeline.yaml` 设置检索与处理参数，然后直接运行：

```bash
python -m src.main
```

默认行为：
- 根据配置执行 Google 搜索
- 抓取结果页面并提取正文，按配置分块
- 使用 LLM 进行分析
- 在 `outputs/` 生成带时间戳的 Markdown 报告

切换到自定义配置文件：

```bash
python -m src.main --config config/my_pipeline.yaml
```

### 关键配置项说明（节选）

- `search.queries`：查询关键词列表
- `search.site / language / country / safe`：站点限制、语言、国家、SafeSearch
- `search.max_results / per_page / top_k_per_query`：抓取数量与分页
- `fetch.timeout / user_agent`：抓取超时与 UA
- `processing.min_text_length / chunk_size / chunk_overlap / max_chunks_total`：最小文本长度、分块大小、重叠、最大分块数
- `llm.model / temperature / system_prompt_path / max_input_chars`：LLM 模型、温度、系统提示词文件、最大输入字符
- `pipeline.output_dir` 与 `report.title / filename_prefix`：输出目录与报告元信息

---

## 调试子命令（可选）

仅进行 Google 检索（便于排查搜索本身）：

```bash
python -m src.main google-search "your query" --num 5 --json
```

支持常用参数：站点限制（`--site`）、精确短语（`--exact`）、排除词（`--exclude`）、语言/国家（`--language`/`--country`）、SafeSearch（`--safe`）等。

---

## 常见问题

- 报错 “Import bs4 could not be resolved”
  - 请确认已安装 `beautifulsoup4`（执行 `python -m pip install -r requirements.txt`）
- Google API 返回配额错误或 429/5xx
  - 稍后重试；本项目默认启用了指数退避的自动重试
- 提取后的正文过短被丢弃
  - 调整 `processing.min_text_length`、尝试更换站点或关键词

---

## 后续增强建议

- 并行接入 Context7 作为另一类检索器（在配置中选择 `retrievers: [google, context7]`）
- 更鲁棒的正文抽取（readability/boilerpipe 类库）
- 去重与相似度筛选，避免重复页面
- 搜索与抓取结果缓存，加速二次运行、减少配额消耗
- 针对提取与分块的单元测试

---

## Part B：两阶段标注与评测（结合 Alignment Report）

前置：你已生成 Alignment Report（Markdown），例如：

```
outputs/alignment_report_20251024_075511/alignment_report_20251024_075511.md
```

数据集：放于 `data/OD_Multilingual/*.xlsx`（例如 `od_data_ori_J.xlsx`）。程序会自动猜测文本列和标签列（OD/ED/SH），也可以用参数覆盖。

运行：

```bash
python -m src.main annotate \
  outputs/alignment_report_20251024_075511/alignment_report_20251024_075511.md \
  data/OD_Multilingual/od_data_ori_J.xlsx \
  --model openai:gpt-5-chat-free \
  --workers 4 \
  --eval
```

说明：
- 阶段1：将原文与 Alignment Report 一并传给 LLM，识别并解释亚文化词汇/表达，产出“现代通用语言”的改写与解释（JSON）。
- 阶段2：在同一会话中继续，让 LLM 基于阶段1的输出做 OD/ED/SH 三维度判定（0/1/2）。
- 输出位于 `outputs/annotations_*/predictions.jsonl`，并保存每条的 LLM 调用日志。
- 若数据中包含金标（OD/ED/SH 列），加 `--eval` 可计算 Macro F1。

合并评测方式：
- 若使用 `--resume-from` 续跑且希望对“上次+本次”的整体结果评测，可加 `--eval --eval-merged`，系统会在评测时将两次预测合并（按 `source_id` 去重，后者覆盖前者）。
- 或使用独立子命令对任意多个预测文件合并评测：

```bash
python -m src.main evaluate \
  data/OD_Multilingual/od_data_ori_J.xlsx \
  outputs/annotations_20251028_000348/predictions.jsonl \
  outputs/annotations_20251028_010101/predictions.jsonl \
  --dedupe-by source_id --keep last --id-key _row
```

性能与控制：
- `--workers N`：并行执行 N 个线程进行 LLM 调用（默认 4）。建议从小数值开始，避免触发平台限流。
- `--limit K`：只处理前 K 条，适合冒烟测试。
- `--resume-from <predictions.jsonl>`：跳过之前已处理的样本（基于 id + 上下文指纹）。
- `--timeout` 与 `--retries`：请求超时与重试次数。
- `--on-error {skip|neutral|raise}`：单条请求失败或被拦截时的处理策略。
