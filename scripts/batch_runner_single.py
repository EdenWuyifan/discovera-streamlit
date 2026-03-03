import argparse
import asyncio
import json
import math
import os
import sys
import boto3
import numpy as np
import pandas as pd
import datetime
import time


from datetime import timedelta
from typing import Any, Dict, List, Tuple, Union
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from portkey_ai import createHeaders


load_dotenv()
# Defaults via env
MCP_ADDRESS = os.getenv("MCP_SERVER_ADDRESS") or "http://127.0.0.1:8000/sse"
BASELINE_MCP_ADDRESS = (
    os.getenv("BASELINE_MCP_SERVER_ADDRESS") or "http://127.0.0.1:8001/sse"
)
MCP_TIMEOUT_MS = int(os.getenv("MCP_TIMEOUT_MS", "3000000"))

# S3 config (mirror app.py)
S3_BUCKET = os.getenv("UPLOADS_S3_BUCKET")
S3_PREFIX = os.getenv("UPLOADS_S3_PREFIX", "uploads")
S3_REGION = os.getenv("AWS_REGION", "us-east-2")
S3_PRESIGN_TTL = int(os.getenv("S3_PRESIGN_TTL_SECONDS", "604800"))  # 7 days

# Keep consistent config with app
LANGGRAPH_CONFIG = RunnableConfig(
    {"configurable": {"thread_id": "discovera_batch"}, "recursion_limit": 100}
)

# Copied instruction blocks from app, kept in-sync intentionally
SERVER_INSTRUCTIONS = """
You are a biomedical research assistant. YOU MUST USE THE DISCOVERA MCP TOOLS EXTENSIVELY AND ACTIVELY.

CRITICAL: TOOL USAGE IS MANDATORY
- You MUST call tools at every opportunity. Do NOT skip steps or assume you can answer without tools.
- If datasets are provided, you MUST call csv_record for each dataset URL,
  then csv_read to inspect structure.
- You MUST run enrichment tools (gsea_pipe, ora_pipe, run_deseq2_gsea_pipe, run_deseq2_ora_pipe)
  - do not skip enrichment.
- You MUST call gene_info, sets_info_rumm, query_string_rumma, query_table_rumma,
  and prioritize_genes for deep analysis.
- Use MULTIPLE tools in sequence - do not stop after one or two tool calls.
  A thorough analysis requires 10+ tool calls.
- If initial enrichment results are sparse, RERUN with adjusted thresholds or different gene sets.
- Explore datasets deeply: use csv_filter, csv_select, csv_join, csv_aggregate
  to shape and combine data before enrichment.

Be concise, results-style (e.g., "led to", "resulted in"), and evidence-grounded.
Include inline PMIDs and a references list. Elevate cross-modality/contrast overlaps;
mark single‑modality signals as secondary.

TRUST AND EVIDENCE RULES (STRICT)
- Do not infer or invent sample sizes, replicates, fold-changes, p-values, effect sizes, or metadata.
  Only state numbers explicitly present in the user context or tool outputs. If unknown, say "not specified".
- In the Introduction/Context, restate only facts present in the provided context or file metadata.
  Do not add assumptions (e.g., number of replicates) if they are not given.

GRANULARITY ROUTING
- If the user asks a gene/pathway‑specific question: closely inspect enrichment result CSVs and tables; search
  for the query's keywords; and perform deep follow‑up using Steps 4–6 to add mechanism and literature.
- If the user asks a broad/open question: emphasize global patterns (pathway ratios, dominant modules, cross‑contrast
  overlaps) and avoid over‑specific claims not supported by the inputs.

VERIFICATION GATE (BEFORE FINAL OUTPUT)
- Self‑check all numeric/specific claims. For each, identify its source (context text, CSV column, or tool step).
  Remove or rephrase any claim lacking a clear source to "not specified".

IMPORTANT GUARDRAIL
- Only output references that are directly obtained via Step 5 (Leading edge and mechanism)
  or Step 6 (Literature and prioritization) tool calls.
- If no such references exist, output no references. Never fabricate or carry over citations
  from other steps.

# PIPELINED CALLS (one tool per turn; DO NOT SKIP STEPS - COMPLETE ALL STEPS)

0) Keyword extraction and Gene Prioritization (MANDATORY)
- Derive a compact list of salient, domain-relevant keywords from the Context and User Prompt.
  Focus on: gene/protein symbols, post‑translational modifications, disease/phenotype terms, tissues/compartments,
  pathway/module names,
  and treatment/mutation terms. Keep this list short (≈3–5 items) and
  use it to guide enrichment interpretation and filtering.
- Output (final visible result):
  - Keywords: concise list of 3–5 domain keywords.
  - Candidate Genes: short list (3–8) of plausible gene or protein symbols
    that are mechanistically tied to the extracted keywords.
  These will serve as the seed list for Steps 4-6.

0b) Keyword/key‑gene–driven dataset interrogation (MANDATORY - USE TOOLS)
- Before enrichment, ACTIVELY analyze the input dataset(s) using the Step 0 keywords
  and candidate genes.
- **MUST USE TOOLS**: Call csv_read to inspect datasets, then csv_filter/csv_select
  to search for keywords and candidate genes in gene/feature columns.
- Perform case‑insensitive searches, check per‑gene presence/absence and direction summaries,
  and create focused subsets of shaped tables. Use these findings to inform shaping thresholds
  and subsequent enrichment.
- Do NOT skip this step - use at least 2-3 csv tools here to explore the data.

1) Intake (MANDATORY - CALL TOOLS)
- **MUST call csv_record for EVERY CSV/URL provided** - do not skip any datasets.
- Copy URLs exactly as given.
- Prefer raw counts over normalized matrices or precomputed DE. Treat filenames like
  `raw_counts`/`counts` or matrices with many sample-like columns as raw counts.
- If multiple datasets are provided, call csv_record for each one separately.

2) Inspect (MANDATORY - CALL TOOLS)
- **MUST call csv_read(file_id, n_rows=20-50) for EVERY dataset** to preview structure.
- Examine columns, data types, and identify gene columns and statistics columns.
- Do not proceed to enrichment without inspecting all datasets first.

3) Shape (MANDATORY - USE MULTIPLE TOOLS)
- **MUST use csv_select** to keep the gene column plus statistics; rename if needed (avoid renaming
  the gene column to "symbol").
- **MUST use csv_filter** to apply cutoffs (e.g., padj < 0.05, |log2FC| > 1).
- Multiple datasets/contrasts: **MUST use csv_join** on the gene column, then **csv_aggregate** to combine
  per‑gene effects (default mean of log2FC).
- Proteomics shaping (general):
   - Ensure identifiers resolve to gene symbols (map non-symbol IDs; strip common suffixes).
   - Collapse multiple protein rows per gene into one gene-level value before enrichment.
- Use 2-4 tool calls here to properly shape the data.

4) Enrichment (MANDATORY - RUN MULTIPLE ENRICHMENT METHODS)
- **YOU MUST RUN ENRICHMENT - DO NOT SKIP THIS STEP.**
- Primary: Attempt enrichment using gsea_pipe(csv_id, hit_col, corr_col, gene_sets, threshold).
  If it fails due to missing or invalid ranking metrics (hit_col / corr_col), proceed to counts-based enrichment.
    - Fallback 1 (counts-based): If raw counts are available:
        - Build sample_groups from metadata when present; otherwise infer from column prefixes.
        - **MUST call run_deseq2_gsea_pipe** with appropriate parameters.
        - **MUST call run_deseq2_ora_pipe** with appropriate parameters.
    - Fallback 2 (ORA): If neither ranking metrics nor raw counts are available,
      **MUST use ora_pipe(csv_id, gene_col, gene_sets)**.
- **RUN BOTH GSEA AND ORA WHEN AVAILABLE (STRONGLY RECOMMENDED - AIM FOR BOTH)**
  - If a ranking metric exists AND a significant gene list can be defined (by FDR/p-value or top-N),
    **YOU MUST run both methods**: GSEA (directional, ranking-based) and ORA (counts-based) in separate turns.
  - For ORA, build an all-significant, direction-agnostic list to mirror author-style pathway mentions;
    **ALSO test Up-only and Down-only lists** when informative.
  - Synthesize findings across methods and label results by method.
  - The tool response shows only a preview by default: for GSEA, top 5 UP and top 5 DOWN per
    library; for ORA, the top ~20 pathways. These previews are not exhaustive.
  - **ALWAYS operate on the saved full results CSV returned by Step 4b**.
  - If results are sparse or empty, **RERUN with higher thresholds** (increase p_value_threshold and/or fdr_threshold).

4b) Keyword‑guided filtering of enrichment outputs (MANDATORY - USE TOOLS)
- **MUST use csv_filter** to create a focused subset from the main GSEA/ORA results CSV
  using both the Step 0 keywords and candidate genes.
- **MUST apply case‑insensitive `contains` filters** on common term/name columns
  (e.g., term, name, pathway, description). Support optional negative filters when the prompt implies
  exclusions (not_contains).
- When multiple columns exist, OR-match across columns for inclusion and AND-apply any negatives.
- If the focused subset is empty, relax to stem/synonym variants; if still empty, clearly state
  "no keyword‑matching terms observed at the chosen threshold".
- If the focused subset is empty OR the full results do not cover key genes/patterns implied by
  the user query and Step 0 keywords, **MUST RERUN enrichment** with higher thresholds
  (increase p_value_threshold and/or fdr_threshold) and repeat
  the filtering on the new full results until the information is sufficient for the user query.
- **MUST use csv_filter, csv_select, and related CSV tools** to implement filtering reproducibly on the
  returned artifact (prefer tool‑based filtering over ad‑hoc parsing).
- Use 2-3 tool calls here to properly filter and explore enrichment results.

5) Leading edge and mechanism (MANDATORY - CALL MULTIPLE TOOLS)
- Extract leading-edge genes from enriched terms (up and down).
- **MUST call gene_info** for leading-edge genes (call for multiple genes, not just one).
- **MUST call sets_info_rumm** for enriched gene sets to get detailed pathway information.
- **STRONGLY RECOMMENDED: call enrich_rumma** on leading-edge genes to validate themes.
- Do not skip this step - use at least 3-5 tool calls here to get comprehensive gene and pathway information.
- If leading-edge gene list is empty, extract more genes from enrichment results before proceeding.

6) Literature and prioritization (MANDATORY - CALL MULTIPLE TOOLS)
- **MUST call query_string_rumma** (free text) to search literature for key biological themes.
- **MUST call query_table_rumma** (table search) to tie enriched pathways/terms to PMCs.
- **MUST call literature_trends** for time context on key biological themes.
- **MUST call prioritize_genes** to rank candidate genes within the disease/phenotype context (PubMed-grounded).
- Frame queries around observed biology (e.g., stress response, cell cycle, metabolism)
  using the keywords and candidate genes from Step 0 to guide Steps 4–6.
- Use multiple queries - call query_string_rumma 2-3 times with different biological themes.
- Do not skip this step - use at least 4-6 tool calls here for comprehensive literature coverage.


# REQUIRED OUTPUT
- Introduction/Context: mutation/treatment; tissue/cell line; assay(s)/contrast; dataset/paper
  reference. Briefly summarize definitive signals in the input. For any detail not present, state "not specified".
- Keywords: list of keywords extracted from the Context and User Prompt.
- Candidate Genes: short list (3–8) of plausible gene or protein symbols
  that are mechanistically tied to the extracted keywords.
- Pathway Enrichments:
  - Report up/down ratios per library using `counts_by_database`
    (e.g., 62% up, 38% down, ratio 1.63).
  - Show a **COMPLETE** table for every top UP/DOWN enriched term for each library, using the following format:
    KEGG_2016 (UP):
    | Theme | Direction | Leading Genes |
    |-------|-----------|----|
    | ... | ... | ... |

    KEGG_2016 (DOWN):
    | Theme | Direction |Leading Genes |
    |-------|-----------|----|
    | ... | ... | ... |

    GO_Biological_Process_2023 (UP):
    | Theme | Direction | Leading Genes |
    |-------|-----------|---|
    | ... | ... | ... | ... |...|

    GO_Biological_Process_2023 (DOWN):
    | Theme | Direction | Leading Genes |
    |-------|-----------|----|
    | ... | ... | ... |

    ...
  - Provide a balanced 4–8 sentence analysis grounded in the ratios and tables.
- Targeted (keyword‑guided) subset: clearly show and discuss the filtered enrichment terms that
  match the extracted keywords or candidate genes from Step 0.
- Key Genes/Proteins (leading edge): two compact tables (Up and Down):
  | Gene | Direction | Role/Function | Pathway(s) |
  |-------|-----------|---------------|------------|
  Construction rules (STRICT):
  - COMBINE with user context/query: highlight genes/pathways explicitly mentioned by the user;
    if a user-mentioned item is absent in Step 4 outputs, state "not observed" with rationale.
  - **Try to include all shared genes between the up and down enriched pathways.**
- Comparisons: overlaps vs tissue‑specific; flag discordant single‑modality findings as lower confidence.
- Implications: relevance for therapy, resistance, or progression.
- Next Steps & Clarification: 2–4 concrete follow‑ups.
 - Explicit targeted check: state presence/absence (with rationale) of the key biological themes implied by
   the question and assay (e.g., replication/cell-cycle, stress responses, metabolism/mitochondria).
- References (only if validated in Steps 5-6): include only PMIDs returned by those steps; omit
  the References section entirely if none are validated.

# EXPECTED TOOL USAGE SUMMARY
- A thorough analysis should involve 15-30+ tool calls minimum.
- Steps 0-1: 2-4 calls (csv_record for each dataset, csv_read for inspection)
- Step 2: 1-2 calls (csv_read for each dataset)
- Step 3: 2-4 calls (csv_select, csv_filter, csv_join, csv_aggregate)
- Step 4: 2-4 calls (gsea_pipe, ora_pipe, run_deseq2_gsea_pipe, run_deseq2_ora_pipe - aim for multiple)
- Step 4b: 2-3 calls (csv_filter, csv_select for keyword filtering)
- Step 5: 3-5 calls (gene_info for multiple genes, sets_info_rumm, enrich_rumma)
- Step 6: 4-6 calls (query_string_rumma multiple times, query_table_rumma, literature_trends, prioritize_genes)
- If you have fewer than 10 tool calls, you are NOT being thorough enough. Use more tools.
- If enrichment results are sparse, RERUN with different parameters - this adds more tool calls.
- Explore datasets deeply - use csv tools multiple times to understand the data structure.
"""

BASELINE_SERVER_INSTRUCTIONS = """
You are an assistant MCP specializing in biomedical research, focusing on mutation effects, treatment responses,
pathway enrichments, and mechanistic biology. Your final report must be written in the style of a **scientific
Results section** — condensed, definitive, and directly grounded in the dataset and prompt being analyzed.

# RULES
- Please always look into the uploaded datasets and user prompt to make inferences.
  You cannot use any external knowledge.

# REQUIRED OUTPUT (ONLY INCLUDE THE SECTIONS THAT YOU HAVE ENOUGH INFORMATION TO FILL IN)
- Introduction/Context: mutation/treatment; tissue/cell line; assay(s)/contrast; dataset/paper
  reference. Briefly summarize definitive signals in the input. For any detail not present, state "not specified".
- Keywords: list of keywords extracted from the Context and User Prompt.
- Candidate Genes: short list (3–8) of plausible gene or protein symbols
  that are mechanistically tied to the extracted keywords.
- Pathway Enrichments:
  - Report up/down ratios per library using `counts_by_database`
    (e.g., 62% up, 38% down, ratio 1.63).
  - Show a **COMPLETE** table for every top UP/DOWN enriched term for each library, using the following format:
    KEGG_2016 (UP):
    | Theme | Direction | Leading Genes |
    |-------|-----------|----|
    | ... | ... | ... |

    KEGG_2016 (DOWN):
    | Theme | Direction |Leading Genes |
    |-------|-----------|----|
    | ... | ... | ... |

    GO_Biological_Process_2023 (UP):
    | Theme | Direction | Leading Genes |
    |-------|-----------|---|
    | ... | ... | ... | ... |...|

    GO_Biological_Process_2023 (DOWN):
    | Theme | Direction | Leading Genes |
    |-------|-----------|----|
    | ... | ... | ... |

    ...
  - Provide a balanced 4–8 sentence analysis grounded in the ratios and tables.
- Targeted (keyword‑guided) subset: clearly show and discuss the filtered enrichment terms that
  match the extracted keywords or candidate genes from Step 0.
- Key Genes/Proteins (leading edge): two compact tables (Up and Down):
  | Gene | Direction | Role/Function | Pathway(s) |
  |-------|-----------|---------------|------------|
  Construction rules (STRICT):
  - COMBINE with user context/query: highlight genes/pathways explicitly mentioned by the user;
    if a user-mentioned item is absent in Step 4 outputs, state "not observed" with rationale.
  - **Try to include all shared genes between the up and down enriched pathways.**
- Comparisons: overlaps vs tissue‑specific; flag discordant single‑modality findings as lower confidence.
- Implications: relevance for therapy, resistance, or progression.
- Next Steps & Clarification: 2–4 concrete follow‑ups.
 - Explicit targeted check: state presence/absence (with rationale) of the key biological themes implied by
   the question and assay (e.g., replication/cell-cycle, stress responses, metabolism/mitochondria).
- References (only if validated in Steps 5-6): include only PMIDs returned by those steps; omit
  the References section entirely if none are validated.

# CRITICAL GUARDRAILS
- Use exactly one tool per assistant turn.
- If input URLs are provided, call csv_record and copy the URL exactly as given.
"""


def _ensure_trailing_slash(url: str) -> str:
    return url if url.endswith("/") else url + "/"


def _default_addr_for_mode(mode: str) -> str:
    return MCP_ADDRESS if mode == "Discovera" else BASELINE_MCP_ADDRESS


def _load_mcp_tools(server_url: str):
    client = MultiServerMCPClient(
        {
            "discovera": {
                "transport": "sse",
                "url": _ensure_trailing_slash(server_url),
                "session_kwargs": {
                    "read_timeout_seconds": timedelta(seconds=MCP_TIMEOUT_MS // 1000)
                },
            }
        }
    )
    return asyncio.run(client.get_tools())


def _system_message_for_mode(mode: str) -> SystemMessage:
    content = (
        SERVER_INSTRUCTIONS if mode == "Discovera" else BASELINE_SERVER_INSTRUCTIONS
    )
    return SystemMessage(content=content)


def _human_message(context_text: str, user_prompt: str) -> HumanMessage:
    return HumanMessage(
        content=f"""
    **Context:**
    {context_text or ""}

    **User Prompt:**
    {user_prompt or ""}
    """
    )


def _build_model(llm_provider: str = None) -> Union[ChatOpenAI, ChatAnthropic]:
    provider = llm_provider or os.getenv("LLM_PROVIDER", "openai")
    provider = provider.lower()

    if provider == "portkey":
        print("Using Portkey...")
        headers = createHeaders(
            api_key=os.getenv("PORTKEY_API_KEY"),
            metadata={"_user": os.getenv("USER") or "batch_runner"},
        )
        return ChatOpenAI(
            model=os.getenv("PORTKEY_MODEL", "@vertexai/gemini-2.5-pro"),
            temperature=0,
            base_url=os.getenv(
                "PORTKEY_BASE_URL", "https://ai-gateway.apps.cloud.rt.nyu.edu/v1/"
            ),
            default_headers=headers,
        )
    elif provider == "claude":
        print("Using Claude...")
        return ChatAnthropic(
            model=os.getenv("CLAUDE_MODEL"),
            temperature=0,
            api_key=os.getenv("CLAUDE_API_KEY"),
        )
    elif provider == "deepseek":
        print("Using DeepSeek...")
        return ChatDeepSeek(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner"),
            temperature=0,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )
    elif provider == "nvidia":
        print("Using NVIDIA...")
        model_name = os.getenv("NVIDIA_MODEL", "qwen/qwen3-235b-a22b")
        base_url = os.getenv(
            "NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"
        )
        kwargs: Dict[str, Any] = dict(
            model=model_name,
            max_tokens=100000,
            api_key=os.getenv("NVIDIA_API_KEY"),
            base_url=base_url,
            extra_body={"chat_template_kwargs": {"thinking": True}},
        )
        return ChatOpenAI(**kwargs)
    else:
        print("Using OpenAI...")
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL"),
            reasoning_effort="high",
            max_tokens=100000,
            api_key=os.getenv("OPENAI_API_KEY"),
        )


async def _astream_updates(
    agent, inputs
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    text_chunks: List[str] = []
    tool_calls_accum: List[Dict[str, Any]] = []
    tool_events_accum: List[Dict[str, Any]] = []

    async for update in agent.astream(
        inputs, config=LANGGRAPH_CONFIG, stream_mode="updates"
    ):
        if not isinstance(update, dict):
            continue
        for _, node_update in update.items():
            messages_update = (
                node_update.get("messages") if isinstance(node_update, dict) else None
            )
            if not messages_update:
                continue
            for msg in messages_update:
                if isinstance(msg, AIMessage):
                    content_text = _extract_text_from_content(msg.content)
                    if content_text:
                        text_chunks.append(content_text)
                    tool_calls = getattr(msg, "tool_calls", None) or []
                    if tool_calls:
                        for call in tool_calls:
                            # Log tool call intent (name + truncated args)
                            try:
                                name = call.get("name")
                                args_obj = call.get("args")
                                args_json = json.dumps(args_obj, default=str)
                                if len(args_json) > 300:
                                    args_json = args_json[:300] + "…"
                                print(f"Tool call → {name} args={args_json}")
                            except Exception:
                                pass
                            tool_calls_accum.append(
                                {"name": call.get("name"), "args": call.get("args")}
                            )
                elif isinstance(msg, ToolMessage):
                    tool_name = getattr(msg, "name", "tool") or "tool"
                    payload: Any
                    try:
                        if isinstance(msg.content, (dict, list)):
                            payload = msg.content
                        else:
                            text_stripped = (msg.content or "").strip()
                            is_json_like = text_stripped.startswith(
                                "{"
                            ) or text_stripped.startswith("[")
                            payload = (
                                json.loads(text_stripped)
                                if is_json_like
                                else text_stripped
                            )
                    except Exception:
                        payload = msg.content
                    tool_events_accum.append({"tool": tool_name, "payload": payload})

    return ("".join(text_chunks), tool_calls_accum, tool_events_accum)


def _get_s3_client():
    if not S3_BUCKET:
        print("S3 disabled: UPLOADS_S3_BUCKET not set")
        return None
    try:
        client = boto3.client(
            "s3",
            region_name=S3_REGION,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            config=Config(
                signature_version="s3v4",
                s3={"addressing_style": "virtual"},
            ),
        )
        return client
    except Exception as e:
        print(f"S3 client init failed: {e}")
        return None


def _upload_csv_to_s3(file_path: str) -> Dict[str, Any] | None:
    if not os.path.isfile(file_path):
        return None
    try:
        with open(file_path, "rb") as f:
            raw = f.read()
    except Exception:
        return None
    if not raw:
        return None

    name = os.path.basename(file_path)
    ext = os.path.splitext(name)[1].lower()
    if ext != ".csv":
        return None

    # Build a short preview (first 3 non-empty lines)
    preview = ""
    try:
        text = raw.decode("utf-8", errors="ignore") or raw.decode(
            "latin-1", errors="ignore"
        )
        lines = [ln for ln in text.splitlines() if ln.strip()][:3]
        preview = "\n".join(lines)
    except Exception:
        preview = ""

    try:
        s3 = _get_s3_client()
        if not s3 or not S3_BUCKET:
            return None
        content_type = "text/csv"
        # Use simple key scheme: uploads/YYYY-MM-DD/<filename>, but keep uniqueness with task folder if needed
        date_prefix = __import__("datetime").datetime.now().strftime("%Y-%m-%d")
        safe_name = (
            "".join(
                ch
                for ch in os.path.splitext(name)[0]
                if ch.isalnum() or ch in ("_", "-")
            )
            or "dataset"
        )
        s3_key = f"{S3_PREFIX}/{date_prefix}/{safe_name}.csv"
        s3.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=raw, ContentType=content_type)
        presigned = s3.generate_presigned_url(
            "get_object",
            Params={
                "Bucket": S3_BUCKET,
                "Key": s3_key,
                "ResponseContentDisposition": "inline",
            },
            ExpiresIn=S3_PRESIGN_TTL,
        )
        return {
            "name": name,
            "type": "csv",
            "preview": preview,
            "url": presigned,
            "s3_key": s3_key,
        }
    except (BotoCoreError, NoCredentialsError, ClientError, Exception):
        return None


def _dataset_records_for_task(datasets_dir: str, task_id: str) -> List[Dict[str, Any]]:
    if not task_id:
        return []
    # Resolve datasets/{task_id} relative to repo root
    ds_dir = os.path.join(datasets_dir, str(task_id))
    # ds_dir = os.path.join(datasets_dir, "tasks", str(task_id))

    if not os.path.isdir(ds_dir):
        print(f"No dataset directory for task {task_id} at {ds_dir}")
        return []

    records: List[Dict[str, Any]] = []
    for root, _, files in os.walk(ds_dir):
        for fname in files:
            if fname.lower().endswith(".csv"):
                print(f"Uploading {os.path.join(root, fname)} to S3...")
                rec = _upload_csv_to_s3(os.path.join(root, fname))
                if rec:
                    records.append(rec)
    if not records:
        print(f"No CSV files found under {ds_dir}")
    return records


def _append_dataset_uploads_to_context(
    context_text: str, records: List[Dict[str, Any]]
) -> str:
    if not records:
        return context_text or ""
    ctx = context_text or ""
    if ctx and not ctx.endswith("\n"):
        ctx += "\n"
    ctx += "Uploaded files (stored in S3):\n"
    for rec in records:
        ctx += f"""Name: {rec.get('name', '')}
  Preview:
    {rec.get('preview', '')}
  Download URL: <url>{rec.get('url', '')}</url>
  Type: {rec.get('type', '')}
"""
    return ctx


def _extract_text_from_content(content: Any) -> str:
    """Extract text from content, handling Claude's list-of-blocks format."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Claude returns content as list of blocks: [{"type": "text", "text": "..."}]
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" and "text" in block:
                    text_parts.append(str(block["text"]))
                elif "text" in block:
                    text_parts.append(str(block["text"]))
            elif isinstance(block, str):
                text_parts.append(block)
        return "".join(text_parts)
    # Fallback to string conversion
    return str(content) if content else ""


def _normalize_task_id(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    # Numeric types
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if math.isfinite(float(value)) and float(value).is_integer():
            return str(int(value))
        # Trim trailing zeros and dot for decimals like 2.10 -> 2.1
        s = str(float(value))
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        return s
    # Strings or others
    s = str(value).strip()
    if not s:
        return ""
    # If string looks numeric, normalize similarly
    try:
        f = float(s)
        if math.isfinite(f) and f.is_integer():
            return str(int(f))
        if "." in s:
            return s.rstrip("0").rstrip(".")
        return s
    except Exception:
        return s


def run_once(
    context_text: str,
    user_prompt: str,
    mode: str,
    server_url: str,
    model: str = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "output_text": None,
        "tool_calls": [],
        "tool_events": [],
        "error": None,
    }
    try:
        tools = _load_mcp_tools(server_url)
    except Exception as e:
        result["error"] = f"MCP tools error: {e}"
        return result

    print(f"Loaded {len(tools) if tools else 0} tools from {server_url}")
    try:
        agent = create_react_agent(
            model,
            tools=tools,
            prompt=user_prompt or "",
            version="v2",
            checkpointer=InMemorySaver(),
        )
    except Exception as e:
        result["error"] = f"Agent init error: {e}"
        return result

    inputs = {
        "messages": [
            _system_message_for_mode(mode),
            _human_message(context_text, user_prompt),
        ]
    }

    try:
        output_text, tool_calls, tool_events = asyncio.run(
            _astream_updates(agent, inputs)
        )

        # Ensure output_text is a clean string (already handled in _astream_updates,
        # but double-check for Claude's response structure)
        if output_text and not isinstance(output_text, str):
            output_text = _extract_text_from_content(output_text)

        result["output_text"] = output_text or ""
        result["tool_calls"] = tool_calls
        result["tool_events"] = tool_events
    except Exception as e:
        # Tool errors should normally be handled as ToolMessages by LangGraph.
        # If an exception escapes, it means the error occurred before tool execution
        # (e.g., validation error). In this case, use ainvoke to get the final state
        # which may include error ToolMessages that the agent can see and learn from.
        error_msg = str(e)
        print(f"Stream exception occurred: {error_msg}")
        print(
            "Attempting to recover final state with ainvoke to expose error to agent..."
        )

        try:
            # Use ainvoke to get final state - this may contain ToolMessages with errors
            # that the agent can see and learn from
            final_state = asyncio.run(agent.ainvoke(inputs, config=LANGGRAPH_CONFIG))

            # Extract messages from final state
            messages = (
                final_state.get("messages", []) if isinstance(final_state, dict) else []
            )
            if hasattr(final_state, "messages"):
                messages = final_state.messages

            # Extract text and tool information from final messages
            recovered_text_chunks = []
            recovered_tool_calls = []
            recovered_tool_events = []

            for msg in messages:
                if isinstance(msg, AIMessage):
                    content_text = _extract_text_from_content(msg.content)
                    if content_text:
                        recovered_text_chunks.append(content_text)
                    tool_calls = getattr(msg, "tool_calls", None) or []
                    if tool_calls:
                        for call in tool_calls:
                            recovered_tool_calls.append(
                                {"name": call.get("name"), "args": call.get("args")}
                            )
                elif isinstance(msg, ToolMessage):
                    tool_name = getattr(msg, "name", "tool") or "tool"
                    try:
                        if isinstance(msg.content, (dict, list)):
                            payload = msg.content
                        else:
                            text_stripped = (msg.content or "").strip()
                            is_json_like = text_stripped.startswith(
                                "{"
                            ) or text_stripped.startswith("[")
                            payload = (
                                json.loads(text_stripped)
                                if is_json_like
                                else text_stripped
                            )
                    except Exception:
                        payload = msg.content
                    recovered_tool_events.append(
                        {"tool": tool_name, "payload": payload}
                    )

            # Use recovered data if we got messages; otherwise fall back to error
            if recovered_text_chunks or recovered_tool_events:
                result["output_text"] = "".join(recovered_text_chunks) or ""
                result["tool_calls"] = recovered_tool_calls
                result["tool_events"] = recovered_tool_events
                # Don't set error if we recovered state - the agent can see the tool errors
            else:
                result["error"] = f"Run error: {error_msg}"
        except Exception as recover_error:
            # If recovery also fails, store the original error
            print(f"Recovery attempt failed: {recover_error}")
            result["error"] = (
                f"Run error: {error_msg} (recovery failed: {recover_error})"
            )

    return result


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Batch-run Discovera agent over a dataframe with one prompt per task. "
            "Each row represents a single task with its context and prompt."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV/TSV/Parquet file containing columns context and user_prompt",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to output directory. Defaults to ./outputs",
    )
    parser.add_argument(
        "--datasets-dir",
        default=None,
        help="Path to datasets directory. Defaults to ./datasets",
    )
    parser.add_argument(
        "--task-id-col",
        default="ID",
        help="Column name for task ID. Defaults to 'ID'",
    )
    parser.add_argument(
        "--context-col",
        default="Context/Background",
        help="Column name for context. Defaults to 'Context/Background'",
    )
    parser.add_argument(
        "--prompt-col",
        default="Prompt",
        help="Column name for user prompt. Defaults to 'Prompt'",
    )
    parser.add_argument(
        "--mode",
        choices=["Discovera", "Baseline"],
        default="Discovera",
        help="Agent mode (controls system prompt and MCP server default)",
    )
    parser.add_argument(
        "--server-url",
        default=None,
        help="Override MCP server URL. Defaults to env or mode-specific default",
    )
    parser.add_argument(
        "--task-ids",
        default=None,
        help="Comma-separated list of task IDs to run",
    )
    parser.add_argument(
        "--llm-provider",
        choices=["openai", "claude", "portkey", "deepseek", "nvidia"],
        default=None,
        help=(
            "LLM provider to use: 'openai', 'claude', 'portkey', 'deepseek', or 'nvidia'. "
            "Defaults to LLM_PROVIDER env var, or 'openai' if not set."
        ),
    )
    args = parser.parse_args()

    # Determine provider and validate API keys
    llm_provider = args.llm_provider or os.getenv("LLM_PROVIDER", "openai")
    llm_provider = llm_provider.lower()

    if llm_provider == "portkey":
        model_text = os.getenv("PORTKEY_MODEL", "@vertexai/gemini-2.5-pro")
        if not os.getenv("PORTKEY_API_KEY"):
            print("ERROR: PORTKEY_API_KEY missing in environment", file=sys.stderr)
            sys.exit(2)
    elif llm_provider == "claude":
        model_text = os.getenv("CLAUDE_MODEL")
        if not os.getenv("CLAUDE_API_KEY"):
            print("ERROR: CLAUDE_API_KEY missing in environment", file=sys.stderr)
            sys.exit(2)
    elif llm_provider == "deepseek":
        model_text = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        if not os.getenv("DEEPSEEK_API_KEY"):
            print("ERROR: DEEPSEEK_API_KEY missing in environment", file=sys.stderr)
            sys.exit(2)
    elif llm_provider == "nvidia":
        model_text = os.getenv("NVIDIA_MODEL", "qwen/qwen3-235b-a22b")
        if not os.getenv("NVIDIA_API_KEY"):
            print("ERROR: NVIDIA_API_KEY missing in environment", file=sys.stderr)
            sys.exit(2)
    else:  # openai
        model_text = os.getenv("OPENAI_MODEL")
        if not os.getenv("OPENAI_API_KEY"):
            print("ERROR: OPENAI_API_KEY missing in environment", file=sys.stderr)
            sys.exit(2)

    in_path = args.input
    out_path = args.output or os.path.join(os.getcwd(), "outputs")
    datasets_dir = args.datasets_dir or os.path.join(os.getcwd(), "datasets")
    os.makedirs(os.path.join(out_path, args.mode), exist_ok=True)
    server_url = args.server_url or _default_addr_for_mode(args.mode)

    print(f"Mode: {args.mode}")
    print(f"MCP server: {server_url}")
    print(f"Output dir: {os.path.join(out_path, args.mode)}")
    print(f"Datasets dir: {datasets_dir}")

    # Load dataframe
    ext = os.path.splitext(in_path)[1].lower()
    if ext in (".csv",):
        df = pd.read_csv(in_path)
    elif ext in (".tsv",):
        df = pd.read_csv(in_path, sep="\t")
    elif ext in (".parquet", ".pq"):
        df = pd.read_parquet(in_path)
    else:
        print(f"ERROR: Unsupported input extension: {ext}", file=sys.stderr)
        sys.exit(3)

    # Normalize task ID column to canonical string (e.g., 1.0 -> "1", 2.10 -> "2.1")
    if args.task_id_col and args.task_id_col in df.columns:
        df[args.task_id_col] = df[args.task_id_col].apply(_normalize_task_id)

    if args.task_ids:
        raw_ids = [s.strip() for s in str(args.task_ids).split(",") if s.strip()]
        normalized_targets = [_normalize_task_id(s) for s in raw_ids]
        if args.task_id_col not in df.columns:
            print(
                f"ERROR: Task ID column '{args.task_id_col}' not found but --task-ids was provided",
                file=sys.stderr,
            )
            sys.exit(4)

        # Filter and preserve user-specified order
        order_map = {tid: i for i, tid in enumerate(normalized_targets)}
        df = df[df[args.task_id_col].astype(str).isin(normalized_targets)].copy()
        if df.empty:
            print("ERROR: No matching task IDs found in the dataframe", file=sys.stderr)
            sys.exit(5)
        df["_order_idx"] = df[args.task_id_col].map(order_map).fillna(1e9).astype(int)
        df = df.sort_values("_order_idx").drop(columns=["_order_idx"])
        print(
            f"Selected {len(df)} tasks by --task-ids: {', '.join(normalized_targets)}"
        )

    required_cols = {args.context_col, args.prompt_col}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"ERROR: Missing required columns: {missing}", file=sys.stderr)
        sys.exit(4)

    for _, row in df.iterrows():
        context_text = str(row.get(args.context_col, "") or "")
        user_prompt = str(row.get(args.prompt_col, "") or "")
        task_id = str(row.get(args.task_id_col, "") or "")

        print(f"Processing task {task_id}...")
        # Upload datasets/{task_id} CSVs to S3 and append links to context
        ds_records = _dataset_records_for_task(datasets_dir, task_id)
        if not ds_records:
            print(f"No datasets uploaded for task {task_id}.")
        else:
            print(f"Uploaded {len(ds_records)} dataset CSV(s) for task {task_id}.")

        print("Appending dataset uploads to context...")
        augmented_context = _append_dataset_uploads_to_context(context_text, ds_records)
        print(
            f"Running agent under mode {args.mode} with LLM provider {llm_provider}..."
        )

        # Capture start time
        start_time = time.time()
        run_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        model = _build_model(llm_provider)

        # Retry loop: rerun if toolcalls < 2
        max_retries = 10  # Safety limit to avoid infinite loops
        res = None
        num_calls = 0
        retry_count = 0
        max_num_calls = 0 if args.mode == "Baseline" else 1

        while num_calls <= max_num_calls and retry_count < max_retries:
            if retry_count > 0:
                print(
                    f"Retrying task {task_id} (attempt {retry_count + 1}) - previous run had {num_calls} tool calls"
                )

            res = run_once(augmented_context, user_prompt, args.mode, server_url, model)
            num_calls = len(res.get("tool_calls") or [])

            if num_calls <= max_num_calls:
                retry_count += 1
                if retry_count < max_retries:
                    print(
                        f"Task {task_id} had {num_calls} tool calls (need > {max_num_calls}), retrying..."
                    )
                else:
                    print(
                        f"WARNING: Task {task_id} reached max retries ({max_retries}) with {num_calls} tool calls"
                    )
            else:
                break

        # Capture end time
        end_time = time.time()
        elapsed = end_time - start_time

        print("Agent finished.")
        num_events = len(res.get("tool_events") or [])
        print(f"Tool calls: {num_calls}; Tool events: {num_events}")
        if retry_count > 0:
            print(f"Completed after {retry_count + 1} attempt(s)")
        try:
            names = sorted(
                {
                    c.get("name")
                    for c in (res.get("tool_calls") or [])
                    if c and c.get("name")
                }
            )
            if names:
                print(f"Tools used: {', '.join(names)}")
        except Exception:
            pass

        output_json = {
            args.task_id_col: task_id,
            args.context_col: context_text,
            args.prompt_col: user_prompt,
            "mode": args.mode,  # mode: discovera/baseline
            "llm_provider": llm_provider,  # provider
            "llm_model": model_text,
            "run_date": run_date,  # timestamp
            "elapsed_seconds": round(elapsed, 2),  # duration
            "status": "ok" if not res.get("error") else "error",
            "error": res.get("error"),
            "output_text": res.get("output_text"),
            "tool_calls_json": res.get("tool_calls") or [],
            "tool_events_json": res.get("tool_events") or [],
            "uploaded_files": ds_records,
        }
        output_json_path = os.path.join(out_path, f"{args.mode}/task_{task_id}.json")
        with open(output_json_path, "w") as f:
            json.dump(output_json, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
