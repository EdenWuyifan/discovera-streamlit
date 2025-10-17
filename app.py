import asyncio
import hashlib
import json
import os
import queue
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List

import boto3
import httpx
import streamlit as st
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
from dotenv import load_dotenv

# from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

# from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from portkey_ai import createHeaders

# Load .env if present
load_dotenv()

# --- Config ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "o4-mini")
MCP_ADDRESS = os.getenv("MCP_SERVER_ADDRESS") or "http://127.0.0.1:8000/sse"
BASELINE_MCP_ADDRESS = (
    os.getenv("BASELINE_MCP_SERVER_ADDRESS") or "http://127.0.0.1:8001/sse"
)
MCP_TIMEOUT_MS = int(os.getenv("MCP_TIMEOUT_MS", "3000000"))
S3_BUCKET = os.getenv("UPLOADS_S3_BUCKET")
S3_PREFIX = os.getenv("UPLOADS_S3_PREFIX", "uploads")
S3_REGION = os.getenv("AWS_REGION", "us-east-2")
S3_PRESIGN_TTL = int(os.getenv("S3_PRESIGN_TTL_SECONDS", "604800"))  # 7 days
LANGGRAPH_CONFIG = RunnableConfig(
    {"configurable": {"thread_id": "discovera_streamlit"}, "recursion_limit": 100}
)
# duckduckgo = DuckDuckGoSearchRun()

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY missing in environment")
    st.stop()

st.set_page_config(page_title="Discovera Chat", page_icon="🧬", layout="centered")

# Widen the sidebar to make MCP/Context panels wider
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        width: 520px !important;
        min-width: 520px !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        width: 520px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Discovera Chat 🧬")

# --- Mode selection & reinit ---


def _default_addr_for_mode(mode: str) -> str:
    return MCP_ADDRESS if mode == "Discovera" else BASELINE_MCP_ADDRESS


def _reinit_session_for_mode(new_mode: str):
    # Reset all conversational/agent state and mode-dependent settings
    st.session_state["messages"] = []
    st.session_state["mcp_connected"] = False
    st.session_state["checkpointer"] = InMemorySaver()
    st.session_state["code_view_msgs"] = set()
    st.session_state["mcp_server"] = _default_addr_for_mode(new_mode)
    # Also reset context to avoid cross-mode leakage
    st.session_state["context"] = ""
    st.session_state["context_uploads"] = []
    st.session_state["context_upload_hashes"] = set()


if "mode" not in st.session_state:
    st.session_state["mode"] = "Discovera"
if "prev_mode" not in st.session_state:
    st.session_state["prev_mode"] = st.session_state["mode"]
if "mcp_server" not in st.session_state:
    st.session_state["mcp_server"] = _default_addr_for_mode(st.session_state["mode"])

with st.sidebar.expander("Mode", expanded=True):
    st.radio(
        "Run Mode",
        ["Discovera", "Baseline"],
        key="mode",
        help="Choose full Discovera agent or Baseline (CSV-only) agent",
        horizontal=True,
    )

# If mode changed, fully reinitialize
if st.session_state["mode"] != st.session_state["prev_mode"]:
    _reinit_session_for_mode(st.session_state["mode"])
    st.session_state["prev_mode"] = st.session_state["mode"]
with st.sidebar.expander("Connection", expanded=True):
    default_addr = _default_addr_for_mode(st.session_state.get("mode", "Discovera"))
    col1, col2 = st.columns([3, 1])
    with col1:
        st.text_input(
            "MCP Server",
            st.session_state.get("mcp_server", default_addr),
            key="mcp_server",
            help="Remote FastMCP base URL",
            disabled=False,
        )
    with col2:
        if st.button("Check MCP", use_container_width=True):
            addr = st.session_state.get("mcp_server") or default_addr
            url = httpx.URL(addr)
            ok = False
            notes: List[str] = []
            try:
                with httpx.Client(timeout=200.0, follow_redirects=True) as http:
                    # 0) Open SSE request in streaming mode and close immediately after status
                    try:
                        sse_headers = {
                            "Accept": "text/event-stream",
                            "Cache-Control": "no-store",
                            "Connection": "keep-alive",
                        }
                        with http.stream("GET", str(url), headers=sse_headers) as r:
                            notes.append(
                                f"STREAM GET {url.raw_path.decode()} -> {r.status_code}"
                            )
                            if r.status_code == 200:
                                ok = True
                    except Exception as e:
                        notes.append(f"STREAM GET {url.raw_path.decode()} -> {e}")

                st.session_state["mcp_connected"] = ok
                status_msg = "ok" if ok else "failed"
                # Show first few notes to avoid long toasts
                detail = " | ".join(notes)
                st.toast(f"MCP check {status_msg}. {detail}")
            except Exception as e:
                st.session_state["mcp_connected"] = False
                st.toast(f"MCP check failed: {e}")

    connected = bool(st.session_state.get("mcp_connected"))
    status_text = "🟢 connected" if connected else "🔴 disconnected"
    addr_text = st.session_state.get("mcp_server") or default_addr
    st.caption(f"Status: {status_text} · {addr_text}")


server_instructions = """
You are a biomedical research assistant. YOU MUST USE THE DISCOVERA MCP TOOLS to answer.
Be concise, results-style (e.g., “led to”, “resulted in”), and evidence-grounded.
Include inline PMIDs and a references list. Elevate cross-modality/contrast overlaps;
mark single‑modality signals as secondary.

TRUST AND EVIDENCE RULES (STRICT)
- Do not infer or invent sample sizes, replicates, fold-changes, p-values, effect sizes, or metadata.
  Only state numbers explicitly present in the user context or tool outputs. If unknown, say “not specified”.
- In the Introduction/Context, restate only facts present in the provided context or file metadata.
  Do not add assumptions (e.g., number of replicates) if they are not given.

GRANULARITY ROUTING
- If the user asks a gene/pathway‑specific question: closely inspect enrichment result CSVs and tables; search
  for the query’s keywords; and perform deep follow‑up using Steps 4–5 to add mechanism and literature.
- If the user asks a broad/open question: emphasize global patterns (pathway ratios, dominant modules, cross‑contrast
  overlaps) and avoid over‑specific claims not supported by the inputs.

VERIFICATION GATE (BEFORE FINAL OUTPUT)
- Self‑check all numeric/specific claims. For each, identify its source (context text, CSV column, or tool step).
  Remove or rephrase any claim lacking a clear source to “not specified”.

IMPORTANT GUARDRAIL
- Only output references that are directly obtained via Step 4 (Leading edge and mechanism)
  or Step 5 (Literature and prioritization) tool calls.
- If no such references exist, output no references. Never fabricate or carry over citations
  from other steps.

# PIPELINED CALLS (one tool per turn; skip unnecessary steps)

0) Keyword extraction (guides downstream filtering)
- Derive a compact list of salient, domain-relevant keywords from the Context and User Prompt.
  Focus on: gene/protein symbols, post‑translational modifications, disease/phenotype terms, tissues/compartments, pathway/module names,
  and treatment/mutation terms.
- Keep this list short (≈3–5 items) and use it to guide enrichment interpretation and filtering.

1) Intake
- If a CSV/URL is provided: call `csv_record` (copy URLs exactly) → file_id.
- Prefer raw counts over normalized matrices or precomputed DE. Treat filenames like
  `raw_counts`/`counts` or matrices with many sample-like columns as raw counts.

2) Inspect
- `csv_read(file_id, n_rows)` to preview structure.

3) Shape (required before enrichment)
- `csv_select` to keep the gene column plus statistics; rename if needed (avoid renaming
  the gene column to "symbol").
- `csv_filter` to apply cutoffs (e.g., padj/log2FC).
- Multiple datasets/contrasts: `csv_join` on the gene column → `csv_aggregate` to combine
  per‑gene effects (default mean of log2FC).
 - Proteomics shaping (general):
   - Ensure identifiers resolve to gene symbols (map non-symbol IDs; strip common suffixes).
   - Collapse multiple protein rows per gene into one gene-level value before enrichment.

4) Enrichment (choose first applicable; fall back if empty/missing)
- If raw counts exist: `run_deseq2_gsea_pipe(raw_counts_csv_id, sample_groups, gene_column,
  gene_sets, threshold)`. Build `sample_groups` from metadata when present; otherwise infer from
  column prefixes.
- Else, if a ranking metric exists: `gsea_pipe(csv_id, hit_col, corr_col, gene_sets, threshold)`.
- Else: `ora_pipe(csv_id, gene_col, gene_sets)`.
- Include broad libraries for GSEA/ORA (GO, Hallmark, KEGG, Reactome). Use `counts_by_database`
  to compute up/down ratios. If GSEA fails due to missing hit/corr, run counts‑based DE+GSEA if
  available, then retry enrichment.
 - Library coverage: include major signaling, cell cycle/repair, and mitochondrial/energy modules across
   GO, Hallmark, KEGG, Reactome, and compartments as appropriate for the assay.
 - Reduce false negatives:
   - If critical biology is suspected but absent at FDR ≤ 0.05, consider exploratory runs at 0.10 and
     confirm/report at 0.05. Also try ORA on up‑regulated leading‑edge or top‑N genes.

3b) Keyword‑guided filtering of enrichment outputs (REQUIRED)
- From the main GSEA/ORA results CSV produced in Step 3, create a focused subset using the keywords
  from Step 0. Apply case‑insensitive `contains` filters on common term/name columns
  (e.g., term, name, pathway, description). Support optional negative filters when the prompt implies
  exclusions (not_contains).
- When multiple columns exist, OR-match across columns for inclusion and AND-apply any negatives.
- If the focused subset is empty, relax to stem/synonym variants; if still empty, clearly state
  “no keyword‑matching terms observed at the chosen threshold”.
- Use `csv_filter`, `csv_select`, and related CSV tools to implement filtering reproducibly on the
  returned artifact (prefer tool‑based filtering over ad‑hoc parsing).

4) Leading edge and mechanism (REQUIRED)
- From enriched terms (up and down), extract leading genes.
- Call `gene_info`, `query_genes`, or `sets_info_rumm` to add mechanistic context.
- Optional cross-check with external enrichment: call `enrich_rumma` on leading-edge gene lists to validate themes.
 - Seed `query_genes` with salient leading-edge genes and use `count_edges` to summarize by nodes/type.

5) Literature and prioritization (REQUIRED)
- Use `query_string_rumma` (free text) and/or `query_table_rumma` (table search) to tie terms to PMCs.
- Use `literature_trends` for time context.
- **Use `prioritize_genes` to rank candidate genes within a disease/phenotype context (grounded in PubMed).**
 - Use targeted queries reflecting the observed biology (e.g., stress responses, cell-cycle, metabolism).


# REQUIRED OUTPUT
- Introduction/Context: mutation/treatment; tissue/cell line; assay(s)/contrast; dataset/paper
  reference. Briefly summarize definitive signals in the input. For any detail not present, state “not specified”.
- Keywords: list of keywords extracted from the Context and User Prompt.
- Pathway Enrichments:
  - Report up/down ratios per library using `counts_by_database` (e.g., 62% up, 38% down, ratio 1.63).
  - Show a **COMPLETE** table for every top UP/DOWN enriched term for each library, using the following format:
    KEGG_2016 (UP):
    | Theme | Direction | NES/OR | FDR |
    |-------|-----------|--------|-----|
    | ... | ... | ... | ... |

    KEGG_2016 (DOWN):
    | Theme | Direction | NES/OR | FDR |
    |-------|-----------|--------|-----|
    | ... | ... | ... | ... |

    GO_Biological_Process_2023 (UP):
    | Theme | Direction | NES/OR | FDR |
    |-------|-----------|--------|-----|
    | ... | ... | ... | ... |

    GO_Biological_Process_2023 (DOWN):
    | Theme | Direction | NES/OR | FDR |
    |-------|-----------|--------|-----|
    | ... | ... | ... | ... |

    ...
  - Provide a balanced 4–8 sentence analysis grounded in the ratios and tables.
- Targeted (keyword‑guided) subset: clearly show and discuss the filtered enrichment terms that
  match the extracted keywords. Note absence explicitly when none pass FDR.
- Key Genes/Proteins (leading edge): two compact tables (Up and Down):
  | Gene | Direction | Role/Function | Pathway(s) |
  |-------|-----------|---------------|------------|
  Construction rules (STRICT):
  - COMBINE with user context/query: highlight genes/pathways explicitly mentioned by the user;
    if a user-mentioned item is absent in Step 4 outputs, state “not observed” with rationale.
  - Do not list genes lacking support from Step 4/5 outputs.
- Comparisons: overlaps vs tissue‑specific; flag discordant single‑modality findings as lower confidence.
- Implications: relevance for therapy, resistance, or progression.
- Next Steps & Clarification: 2–4 concrete follow‑ups.
 - Explicit targeted check: state presence/absence (with rationale) of the key biological themes implied by
   the question and assay (e.g., replication/cell-cycle, stress responses, metabolism/mitochondria).
- References (only if validated in Steps 4–5): include only PMIDs returned by those steps; omit
  the References section entirely if none are validated.
"""


baseline_server_instructions = """
You are an assistant MCP specializing in biomedical research, focusing on mutation effects, treatment responses,
pathway enrichments, and mechanistic biology. Your final report must be written in the style of a **scientific
Results section** — condensed, definitive, and directly grounded in the dataset and paper being analyzed.

# STYLE & GOALS
- Voice: results-style, concise, causal, definitive (e.g., “led to”, “resulted in”, “confirmed”).

# CRITICAL GUARDRAILS
- Use exactly one tool per assistant turn.
- If input URLs are provided, call csv_record and copy the URL exactly as given.

"""


with st.sidebar.expander("Context", expanded=True):
    st.text_area(
        "Context (sent as instructions every turn)",
        key="context",
        height=120,
        placeholder=(
            "Add study context, constraints, datasets, or goals here.\n"
            "Example: Analyze RNA-seq DE results for lung tissue; prioritize inflammation pathways."
        ),
    )

    # File uploads for CSV/TXT to include as context
    uploaded_files = st.file_uploader(
        "Upload CSV files to include in context",
        type=["csv"],
        accept_multiple_files=True,
        key="context_upload_files",
        help="Files are uploaded to S3 and summarized into the context.",
    )

    # Initialize session state trackers
    if "context_uploads" not in st.session_state:
        st.session_state["context_uploads"] = (
            []
        )  # list of {hash, name, type, preview, url}
    if "context_upload_hashes" not in st.session_state:
        st.session_state["context_upload_hashes"] = set()

    _s3_client = None

    def _get_s3():
        global _s3_client
        if _s3_client is not None:
            return _s3_client
        if not S3_BUCKET:
            return None
        try:
            _s3_client_local = boto3.client(
                "s3",
                region_name=S3_REGION,
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                config=Config(
                    signature_version="s3v4",
                    s3={
                        "addressing_style": "virtual",
                    },
                ),
            )
            # Light touch to validate creds lazily; skip actual call to avoid latency
            _s3_client = _s3_client_local
            return _s3_client
        except Exception:
            return None

    # Local filesystem helpers removed; uploads go directly to S3

    def _save_uploaded_file(file_obj) -> Dict[str, Any] | None:
        try:
            raw: bytes = file_obj.getvalue()
        except Exception:
            return None
        if not raw:
            return None
        sha = hashlib.sha256(raw).hexdigest()
        if sha in st.session_state["context_upload_hashes"]:
            return None  # already processed in this session

        ext = os.path.splitext(file_obj.name)[1].lower() or ".bin"
        # note: timestamp not needed without local writes
        safe_name = os.path.splitext(os.path.basename(file_obj.name))[0]
        safe_name = (
            "".join(ch for ch in safe_name if ch.isalnum() or ch in ("_", "-"))
            or "upload"
        )
        # No local write; upload directly to S3

        # Build a short preview
        preview = ""
        try:
            # Try utf-8 first, fallback to latin-1
            text = raw.decode("utf-8", errors="ignore")
            if not text:
                text = raw.decode("latin-1", errors="ignore")
            if ext == ".txt":
                preview = text[:500]
            elif ext == ".csv":
                # show first 3 non-empty lines
                lines = [ln for ln in text.splitlines() if ln.strip()][:3]
                preview = "\n".join(lines)
            else:
                preview = text[:200]
        except Exception:
            preview = ""

        # Attempt S3 upload with presigned URL generation
        remote_url: str | None = None
        try:
            s3 = _get_s3()
            if s3 and S3_BUCKET:
                # Choose content type
                if ext == ".csv":
                    content_type = "text/csv"
                elif ext == ".txt":
                    content_type = "text/plain"
                else:
                    content_type = "application/octet-stream"

                s3_key = f"{S3_PREFIX}/{datetime.now().strftime('%Y-%m-%d')}/{sha[:12]}_{safe_name}{ext}"
                put_kwargs = {
                    "Bucket": S3_BUCKET,
                    "Key": s3_key,
                    "Body": raw,
                    "ContentType": content_type,
                }
                s3.put_object(**put_kwargs)
                remote_url = s3.generate_presigned_url(
                    "get_object",
                    Params={
                        "Bucket": S3_BUCKET,
                        "Key": s3_key,
                        "ResponseContentDisposition": "inline",
                    },
                    ExpiresIn=S3_PRESIGN_TTL,
                )
        except (BotoCoreError, NoCredentialsError, ClientError, Exception):
            remote_url = None

        if not remote_url:
            return None

        rec = {
            "hash": sha,
            "name": file_obj.name,
            "type": (ext[1:] if ext.startswith(".") else ext).lower(),
            "preview": preview,
            "url": remote_url,
        }
        st.session_state["context_upload_hashes"].add(sha)
        st.session_state["context_uploads"].append(rec)
        return rec

    # Reconcile removals from the uploader UI (handle clicks on the "x")
    current_hashes_in_widget: set[str] = set()
    current_names_in_widget: set[str] = set()
    if uploaded_files:
        for uf in uploaded_files:
            try:
                raw_now: bytes = uf.getvalue()
            except Exception:
                raw_now = b""
            name_now = ""
            try:
                name_now = getattr(uf, "name", "") or ""
            except Exception:
                name_now = ""
            if raw_now:
                sha_now = hashlib.sha256(raw_now).hexdigest()
                current_hashes_in_widget.add(sha_now)
            if name_now:
                current_names_in_widget.add(name_now)
    else:
        # None selected in widget; clear all tracked uploads
        st.session_state["context_uploads"] = []
        st.session_state["context_upload_hashes"] = set()

    if current_hashes_in_widget or current_names_in_widget:
        # Filter existing records by present hashes or names
        existing_uploads = st.session_state.get("context_uploads") or []
        filtered_uploads: List[Dict[str, Any]] = []
        filtered_hashes: set[str] = set()
        for rec in existing_uploads:
            rec_hash = rec.get("hash") or ""
            rec_name = rec.get("name") or ""
            keep_by_hash = bool(rec_hash and rec_hash in current_hashes_in_widget)
            keep_by_name = bool(rec_name and rec_name in current_names_in_widget)
            if keep_by_hash or keep_by_name:
                filtered_uploads.append(rec)
                if rec_hash:
                    filtered_hashes.add(rec_hash)
        st.session_state["context_uploads"] = filtered_uploads
        st.session_state["context_upload_hashes"] = filtered_hashes

    # Process uploads and append to context (deduplicated by content hash)
    new_records: List[Dict[str, Any]] = []
    if uploaded_files:
        for uf in uploaded_files:
            rec = _save_uploaded_file(uf)
            if rec:
                new_records.append(rec)
        if new_records:
            st.write("Uploaded files:")
            for rec in new_records:
                st.markdown(f"- `{rec['name']}` → [download link]({rec['url']})")

# Initialize output files tracker (for tool-returned metadata with s3_presigned_url)
if "output_files" not in st.session_state:
    st.session_state["output_files"] = (
        []
    )  # list of {name, url, size_bytes, created_at, category, origin_tool}
if "output_file_urls" not in st.session_state:
    st.session_state["output_file_urls"] = set()

# Create a placeholder so we can live-update Outputs during streaming
outputs_placeholder = st.sidebar.empty()


def render_outputs():
    with outputs_placeholder.container():
        with st.expander("Outputs", expanded=True):
            _outputs = st.session_state.get("output_files") or []
            if _outputs:
                for _rec in _outputs:
                    _name = _rec.get("name") or _rec.get("s3_key") or "download"
                    _url = _rec.get("url") or _rec.get("s3_presigned_url") or ""
                    if _url:
                        st.markdown(f"- [{_name}]({_url})")
            else:
                st.caption("No outputs yet.")


# Initial render
render_outputs()

st.divider()

# --- Chat state ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []  # UI transcript: {role, content}
if "checkpointer" not in st.session_state:
    print("Creating checkpointer...")
    st.session_state["checkpointer"] = InMemorySaver()

# Show transcript (supports foldable tool entries)


for idx, m in enumerate(st.session_state["messages"]):
    role = m.get("role", "assistant")
    content = m.get("content", "")
    is_foldable = m.get("foldable", False)
    title = m.get("title") or "Details"
    with st.chat_message(role):
        if is_foldable:
            with st.expander(title, expanded=False):
                if role == "assistant":
                    st.code(str(content), language="text")
                else:
                    st.markdown(content)
        else:
            if role == "assistant":
                if "code_view_msgs" not in st.session_state:
                    st.session_state["code_view_msgs"] = set()
                code_view = st.session_state.get("code_view_msgs") or set()
                showing_code = idx in code_view
                if showing_code:
                    st.code(str(content), language="text")
                    if st.button("Back to text", key=f"btn_to_text_{idx}"):
                        code_view.discard(idx)
                        st.session_state["code_view_msgs"] = code_view
                else:
                    st.markdown(content)
                    if st.button("Copy view", key=f"btn_to_code_{idx}"):
                        code_view.add(idx)
                        st.session_state["code_view_msgs"] = code_view
            else:
                st.markdown(content)

# Compose
prompt = st.chat_input("Ask anything…")


# Helper to stream Responses API events
def as_msg(role: str, text: str) -> Dict[str, Any]:
    part_type = "input_text" if role == "user" else "output_text"
    return {"role": role, "content": [{"type": part_type, "text": text}]}


def _ensure_trailing_slash(url: str) -> str:
    return url if url.endswith("/") else url + "/"


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


def _system_message() -> SystemMessage:
    mode = st.session_state.get("mode", "Discovera")
    content = (
        server_instructions if mode == "Discovera" else baseline_server_instructions
    )
    return SystemMessage(content=content)


def _context() -> str:
    ctx = st.session_state.get("context") or ""
    uploads = st.session_state.get("context_uploads") or []
    if uploads:
        ctx += "Uploaded files (stored in S3):\n"
        for rec in uploads:
            ctx += f"""
Name: {rec['name']}
  Preview:
    {rec['preview']}
  Download URL: <url>{rec['url']}</url>
  Type: {rec['type']}
"""
    return ctx


def _human_message(prompt: str) -> HumanMessage:
    ctx = _context()

    print(f"Context: {ctx}")

    return HumanMessage(
        content=f"""
    **Context:**
    {ctx}

    **User Prompt:**
    {prompt}
    """
    )


# @tool
# def duckduckgo_search(query: str) -> str:
#     """Search DuckDuckGo for recent results."""
#     return duckduckgo.run(query)


def _collect_outputs(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.endswith("_metadata") and isinstance(v, dict):
                url = v.get("s3_presigned_url")
                if url:
                    seen = st.session_state.get("output_file_urls") or set()
                    if url not in seen:
                        rec = {
                            "name": v.get("name") or v.get("s3_key") or "download",
                            "url": url,
                            "size_bytes": v.get("size_bytes"),
                            "created_at": v.get("created_at"),
                            "category": v.get("category"),
                            "origin_tool": v.get("origin_tool"),
                            "s3_key": v.get("s3_key"),
                        }
                        out_list = st.session_state.get("output_files") or []
                        out_list.append(rec)
                        st.session_state["output_files"] = out_list
                        seen.add(url)
                        st.session_state["output_file_urls"] = seen
                        render_outputs()


def stream_langgraph(prompt: str):
    mode = st.session_state.get("mode", "Discovera")
    default_addr = _default_addr_for_mode(mode)
    server_url = st.session_state.get("mcp_server") or default_addr

    # Build tools and agent
    try:
        # tools = _load_mcp_tools(server_url) + [duckduckgo_search]
        tools = _load_mcp_tools(server_url)
    except Exception as e:
        yield ("error", f"[MCP tools error] {e}")
        return

    try:
        # model = f"openai:{OPENAI_MODEL}"
        if os.getenv("LLM_PROVIDER") == "portkey":
            portkey_headers = createHeaders(
                api_key=os.getenv("PORTKEY_API_KEY"),
                virtual_key=os.getenv("PROVIDER_API_KEY"),
                metadata={"_user": "yfw215"},
            )
            model = ChatOpenAI(
                model="gemini-2.5-pro",
                temperature=0,
                base_url="https://ai-gateway.apps.cloud.rt.nyu.edu/v1/",
                default_headers=portkey_headers,
            )
        else:
            model = f"openai:{OPENAI_MODEL}"
        agent = create_react_agent(
            model,
            tools=tools,
            prompt=prompt,
            version="v2",
            checkpointer=st.session_state["checkpointer"],
        )
    except Exception as e:
        yield ("error", f"[Agent init error] {e}")
        return

    inputs = {"messages": [_system_message(), _human_message(prompt)]}

    # Stream updates from the graph (agent thoughts + tool calls/results)
    def _stream_agent_updates():
        q: "queue.Queue[Any]" = queue.Queue()

        async def _produce():
            try:
                async for update in agent.astream(
                    inputs, config=LANGGRAPH_CONFIG, stream_mode="updates"
                ):
                    q.put(update)
            except Exception as exc:  # surface errors back to main thread
                q.put(("__error__", exc))
            finally:
                q.put(("__done__", None))

        def _runner():
            asyncio.run(_produce())

        t = threading.Thread(target=_runner, daemon=True)
        t.start()

        while True:
            item = q.get()
            if isinstance(item, tuple) and item[0] == "__error__":
                raise item[1]
            if isinstance(item, tuple) and item[0] == "__done__":
                break
            yield item

    try:
        for update in _stream_agent_updates():
            if not isinstance(update, dict):
                continue
            for node_name, node_update in update.items():
                messages_update = (
                    node_update.get("messages")
                    if isinstance(node_update, dict)
                    else None
                )
                if not messages_update:
                    continue
                for msg in messages_update:
                    # Agent messages ("thoughts"/content)
                    if isinstance(msg, AIMessage):
                        content_text = (
                            msg.content
                            if isinstance(msg.content, str)
                            else str(msg.content)
                        )
                        if content_text:
                            yield ("delta", content_text)
                        # Surface tool call intents
                        tool_calls = getattr(msg, "tool_calls", None) or []
                        if tool_calls:
                            calls_render = []
                            for call in tool_calls:
                                name = call.get("name")
                                args = call.get("args")
                                calls_render.append({"name": name, "args": args})
                            tool_calls_json = json.dumps(
                                calls_render, default=str, indent=2
                            )
                            preview_json = (
                                tool_calls_json
                                if len(tool_calls_json) <= 2000
                                else tool_calls_json[:2000] + "…"
                            )
                            yield (
                                "tool",
                                "Tool calls:\n```json\n" + preview_json + "\n```",
                            )
                    # Tool results
                    elif isinstance(msg, ToolMessage):
                        tool_name = getattr(msg, "name", "tool") or "tool"
                        tool_text = (
                            msg.content
                            if isinstance(msg.content, str)
                            else str(msg.content)
                        )
                        # Do not clip errors: detect via additional kwargs and text heuristics
                        is_error = False
                        try:
                            extra = getattr(msg, "additional_kwargs", {}) or {}
                            is_error = bool(
                                extra.get("is_error")
                                or extra.get("error")
                                or (extra.get("status") == "error")
                            )
                        except Exception:
                            is_error = False
                        if not is_error:
                            lower_text = tool_text.lower()
                            if (
                                "traceback" in lower_text
                                or "exception" in lower_text
                                or "error" in lower_text
                            ):
                                is_error = True

                        # Pretty-print JSON-like tool outputs fully (no truncation)
                        parsed_json = None
                        if not is_error:
                            try:
                                if isinstance(msg.content, (dict, list)):
                                    parsed_json = msg.content
                                else:
                                    text_stripped = tool_text.strip()
                                    if text_stripped.startswith(
                                        "{"
                                    ) or text_stripped.startswith("["):
                                        parsed_json = json.loads(text_stripped)
                            except Exception:
                                parsed_json = None

                        content_to_show = None
                        if parsed_json is not None:
                            _collect_outputs(parsed_json)
                            try:
                                pretty_json = json.dumps(
                                    parsed_json, indent=2, ensure_ascii=False
                                )
                                content_to_show = pretty_json
                            except Exception:
                                content_to_show = None

                        if content_to_show is None:
                            # Fallback to original behavior with clipping for non-JSON or errors
                            content_to_show = (
                                tool_text
                                if is_error or len(tool_text) <= 1000
                                else (tool_text[:1000] + "…")
                            )
                        title_prefix = (
                            f"{tool_name} error" if is_error else f"{tool_name} result"
                        )
                        if parsed_json is not None and not is_error:
                            yield (
                                "tool",
                                f"{title_prefix}:\n```json\n{content_to_show}\n```",
                            )
                        else:
                            yield (
                                "tool",
                                f"{title_prefix}:\n```\n{content_to_show}\n```",
                            )
        # Signal completion
        yield ("done", None)
    except Exception as e:
        yield ("error", f"[Run error] {e}")


if prompt:
    # 1) append user to UI transcript and show it
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2) stream from LangGraph agent
    # 2) stream from LangGraph agent, rendering events in time order (no placeholder)
    for kind, payload in stream_langgraph(prompt):
        if kind == "delta" and payload:
            with st.chat_message("assistant"):
                st.markdown(payload)
            st.session_state["messages"].append(
                {"role": "assistant", "content": payload}
            )
        elif kind == "tool" and payload:
            title_line = (
                str(payload).splitlines()[0].strip() if payload else "Tool event"
            )
            if not title_line:
                title_line = "Tool event"
            with st.chat_message("assistant"):
                with st.expander(title_line[:120], expanded=False):
                    st.markdown(str(payload))
            st.session_state["messages"].append(
                {
                    "role": "assistant",
                    "content": str(payload),
                    "foldable": True,
                    "title": title_line[:120],
                }
            )
        elif kind == "error" and payload:
            with st.chat_message("assistant"):
                st.markdown(str(payload))
            st.session_state["messages"].append(
                {"role": "assistant", "content": str(payload)}
            )
