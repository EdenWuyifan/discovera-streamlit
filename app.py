import os
import json
import asyncio
import threading
import queue
from typing import List, Dict, Any
from datetime import datetime
import hashlib

import streamlit as st
import httpx
from dotenv import load_dotenv
from botocore.config import Config
from botocore.exceptions import BotoCoreError, NoCredentialsError, ClientError
import boto3
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import AIMessage, ToolMessage, SystemMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig

# Load .env if present
load_dotenv()

# --- Config ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MCP_ADDRESS = os.getenv("MCP_SERVER_ADDRESS") or "http://127.0.0.1:8000/sse"
MCP_TIMEOUT_MS = int(os.getenv("MCP_TIMEOUT_MS", "3500"))
S3_BUCKET = os.getenv("UPLOADS_S3_BUCKET")
S3_PREFIX = os.getenv("UPLOADS_S3_PREFIX", "uploads")
S3_REGION = os.getenv("AWS_REGION", "us-east-2")
S3_PRESIGN_TTL = int(os.getenv("S3_PRESIGN_TTL_SECONDS", "604800"))  # 7 days
LANGGRAPH_CONFIG = RunnableConfig(
    {"configurable": {"thread_id": "discovera_streamlit"}}
)

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
with st.sidebar.expander("Connection", expanded=True):
    col1, col2 = st.columns([3, 1])
    with col1:
        st.text_input(
            "MCP Server",
            MCP_ADDRESS,
            key="mcp_server",
            help="Remote FastMCP base URL",
            disabled=False,
        )
    with col2:
        if st.button("Check MCP", use_container_width=True):
            addr = st.session_state.get("mcp_server") or MCP_ADDRESS
            url = httpx.URL(addr)
            ok = False
            notes: List[str] = []
            try:
                with httpx.Client(
                    timeout=MCP_TIMEOUT_MS / 1000.0, follow_redirects=True
                ) as http:
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
    addr_text = st.session_state.get("mcp_server") or MCP_ADDRESS
    st.caption(f"Status: {status_text} · {addr_text}")


server_instructions = """
You are an assistant MCP specializing in biomedical research, focusing on mutation effects, treatment responses,
pathway enrichments, and mechanistic biology. Your final report must be written in the style of a **scientific
Results section** — condensed, definitive, and directly grounded in the dataset and paper being analyzed.

Evidence and confidence:

- Assign confidence scores (high, medium, low) for each claim.
- Specify whether evidence is direct (experimental) or indirect (literature-based).
- Always cite PubMed IDs or database references inline.

---

### Workflow Principles

- Do **not** call every tool in the pipeline by default.  
- Select **only the tools necessary** for the dataset type and research question (e.g., use gsea_pipe if enrichment is needed, gene_info if gene details are critical, query_genes for interactions, etc.).  
- Literature and external knowledge tools (query_string_rumma, query_table_rumma, literature_trends) are **always required** before finalizing.  
- Prefer the minimal set of tools that still produces a mechanistically sound, evidence-supported report.  

---

### Final Report Template (condensed results style)

#### Introduction / Context
- Mutation(s) or treatment studied.  
- Tissue, tumor type, or cell line.  
- Analysis method (e.g., RNA-seq, proteomics, methylation).  
- Reference the dataset/paper explicitly.  

#### Pathway Enrichments
- State the **most significant upregulated and downregulated pathways**, grouped by biological themes.  
- Report significance (NES, FDR, adjusted p-values).  

#### Key Genes / Proteins
- Highlight **leading-edge genes** with their regulation direction and function.  
- Group genes by biological process (e.g., helicase complex, checkpoint regulators, HDR factors).  

#### Mechanistic Interpretation
- Provide a logical **causal explanation** connecting mutations/treatments → pathway changes → molecular mechanisms.  

#### Comparisons
- Mention **tissue-specific vs. overlapping effects** if applicable.  

#### Implications
- State **clinical or therapeutic relevance** definitively.  

#### References
- Inline PubMed IDs for every mechanistic claim.  
- End with a numbered reference list.

---

### Output guidelines (critical):

- Reports must read like a **results section**: definitive, concise, mechanistic.  
- Always tie findings back to the dataset and paper context.  
- Use **direct causal phrasing**: “led to,” “resulted in,” “confirmed.”  
- Mention statistical thresholds (FDR, p-values) when available.  
- Always run at least one literature tool (query_rumma / literature_trends) before finalizing.  
- Do not include unnecessary tool calls — only what’s needed to support the dataset.
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
        "Upload CSV/TXT files to include in context",
        type=["csv", "txt"],
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
            }
        }
    )
    return asyncio.run(client.get_tools())


def _system_message() -> SystemMessage:
    return SystemMessage(content=server_instructions)


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


def stream_langgraph(prompt: str):
    server_url = st.session_state.get("mcp_server") or MCP_ADDRESS

    # Build tools and agent
    try:
        tools = _load_mcp_tools(server_url)
    except Exception as e:
        yield ("error", f"[MCP tools error] {e}")
        return

    try:
        model_id = f"openai:{OPENAI_MODEL}"
        agent = create_react_agent(
            model_id,
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
                        preview = (
                            tool_text
                            if len(tool_text) <= 1000
                            else (tool_text[:1000] + "…")
                        )
                        yield (
                            "tool",
                            f"{tool_name} result:\n```\n{preview}\n```",
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
