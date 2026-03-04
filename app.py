import asyncio
import hashlib
import json
import os
import queue
import threading
import textwrap
from datetime import datetime, timedelta
from typing import Any, Dict, List

import httpx
import streamlit as st
from dotenv import load_dotenv

from web_search import web_search

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from portkey_ai import createHeaders

# Load .env if present
load_dotenv()

# --- Config ---
LLM_PROVIDER = (os.getenv("LLM_PROVIDER") or "openai").strip().lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-3-5")
PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")
PORTKEY_MODEL = os.getenv("PORTKEY_MODEL", "@gpt-5-mini/gpt-5-mini")
PORTKEY_BASE_URL = os.getenv(
    "PORTKEY_BASE_URL", "https://ai-gateway.apps.cloud.rt.nyu.edu/v1/"
)
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "deepseek-ai/deepseek-v3.2")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
MCP_ADDRESS = os.getenv("MCP_SERVER_ADDRESS") or "http://127.0.0.1:8000/sse"
BASELINE_MCP_ADDRESS = (
    os.getenv("BASELINE_MCP_SERVER_ADDRESS") or "http://127.0.0.1:8001/sse"
)
MCP_TIMEOUT_MS = int(os.getenv("MCP_TIMEOUT_MS", "3000000"))
FILE_SERVER_BASE_URL = os.getenv(
    "FILE_SERVER_BASE_URL", "http://127.0.0.1:8001"
).rstrip("/")
FILE_SERVER_TIMEOUT_SECONDS = int(os.getenv("FILE_SERVER_TIMEOUT_SECONDS", "30"))
LANGGRAPH_CONFIG = RunnableConfig(
    {"configurable": {"thread_id": "discovera_streamlit"}, "recursion_limit": 100}
)
# duckduckgo = DuckDuckGoSearchRun()

SUPPORTED_LLM_PROVIDERS = ("openai", "claude", "portkey", "nvidia")
DEFAULT_LLM_PROVIDER = (
    LLM_PROVIDER if LLM_PROVIDER in SUPPORTED_LLM_PROVIDERS else "openai"
)
MODEL_OTHER_OPTION = "Other (custom)"
MODEL_PRESETS: Dict[str, List[str]] = {
    "nvidia": [
        "deepseek-ai/deepseek-v3.2",
        "qwen/qwen3-235b-a22b",
        "z-ai/glm5",
    ],
    "claude": ["claude-haiku-3-5"],
    "portkey": ["@gpt-5-mini/gpt-5-mini"],
    "openai": ["gpt-5-nano"],
}
MODEL_STATE_KEYS = {
    "openai": "openai_model",
    "claude": "claude_model",
    "portkey": "portkey_model",
    "nvidia": "nvidia_model",
}
API_STATE_KEYS = {
    "openai": "openai_api_key",
    "claude": "claude_api_key",
    "portkey": "portkey_api_key",
    "nvidia": "nvidia_api_key",
}


def _current_llm_provider() -> str:
    provider = (
        st.session_state.get("llm_provider_selected")
        or DEFAULT_LLM_PROVIDER
        or "openai"
    )
    provider = str(provider).strip().lower()
    if provider not in SUPPORTED_LLM_PROVIDERS:
        return "openai"
    return provider


def _model_state_key(provider: str) -> str:
    return MODEL_STATE_KEYS.get(provider, "openai_model")


def _api_key_state_key(provider: str) -> str:
    return API_STATE_KEYS.get(provider, "openai_api_key")


def _active_model_name() -> str:
    provider = _current_llm_provider()
    if provider == "portkey":
        return st.session_state.get("portkey_model") or PORTKEY_MODEL
    if provider == "claude":
        return st.session_state.get("claude_model") or CLAUDE_MODEL or ""
    if provider == "nvidia":
        return st.session_state.get("nvidia_model") or NVIDIA_MODEL
    return st.session_state.get("openai_model") or OPENAI_MODEL


st.set_page_config(page_title="Discovera Chat", page_icon="🧬", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
      max-width: 1180px;
      padding-top: 1.1rem;
      padding-bottom: 2rem;
    }
    [data-testid="stSidebarResizer"],
    [data-testid="stSidebarResizeHandle"] {
      display: none !important;
      width: 0 !important;
    }
    div[data-testid="stChatMessage"] {
      border: none !important;
      box-shadow: none !important;
      background: transparent !important;
      margin-bottom: 0.55rem;
    }
    [data-testid="stExpander"] {
      border: none !important;
      box-shadow: none !important;
      background: transparent !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Discovera Chat 🧬")

# --- Mode selection & reinit ---


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


def _serialize_tools(tools: List[Any]) -> List[Dict[str, Any]]:
    infos: List[Dict[str, Any]] = []
    for t in tools or []:
        name = getattr(t, "name", None) or "tool"
        description = getattr(t, "description", None) or ""
        schema_json: str | None = None
        try:
            args_schema = getattr(t, "args_schema", None)
            if args_schema is not None:
                schema_json = args_schema.schema_json(indent=2)
        except Exception:
            schema_json = None
        if schema_json is None:
            try:
                args_obj = getattr(t, "args", None)
                if args_obj is not None:
                    schema_json = json.dumps(args_obj, indent=2, default=str)
            except Exception:
                schema_json = None
        infos.append(
            {
                "name": name,
                "description": description,
                "schema_json": schema_json,
            }
        )
    return infos


def _default_addr_for_mode(mode: str) -> str:
    return MCP_ADDRESS if mode == "Discovera" else BASELINE_MCP_ADDRESS


def _reset_chat_state():
    st.session_state["messages"] = []
    st.session_state["checkpointer"] = InMemorySaver()
    st.session_state["tool_calls_log"] = []
    st.session_state["tool_events_log"] = []
    st.session_state["turn_runs"] = []
    st.session_state["last_error"] = None
    st.session_state["last_status"] = "idle"
    st.session_state["session_started_at"] = datetime.now().isoformat()
    st.session_state["output_files"] = []
    st.session_state["output_file_urls"] = set()


def _latest_user_prompt() -> str:
    messages = st.session_state.get("messages") or []
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return str(msg.get("content") or "")
    return ""


def _latest_output_text() -> str:
    runs = st.session_state.get("turn_runs") or []
    for run in reversed(runs):
        txt = run.get("output_text")
        if txt:
            return str(txt)
    messages = st.session_state.get("messages") or []
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and not msg.get("foldable"):
            return str(msg.get("content") or "")
    return ""


def _session_elapsed_seconds() -> float:
    started = st.session_state.get("session_started_at")
    if not started:
        return 0.0
    try:
        start_dt = datetime.fromisoformat(str(started))
    except Exception:
        return 0.0
    return max((datetime.now() - start_dt).total_seconds(), 0.0)


def _stringify(obj: Any, max_chars: int | None = None) -> str:
    if isinstance(obj, str):
        text = obj
    else:
        try:
            text = json.dumps(obj, indent=2, ensure_ascii=False, default=str)
        except Exception:
            text = str(obj)
    if max_chars is not None and len(text) > max_chars:
        return text[:max_chars] + "\n...[truncated]..."
    return text


def _context_for_export() -> str:
    ctx = st.session_state.get("context") or ""
    uploads = st.session_state.get("context_uploads") or []
    if uploads:
        if ctx and not ctx.endswith("\n"):
            ctx += "\n"
        ctx += "Uploaded files (stored in local file server):\n"
        for rec in uploads:
            ctx += f"""Name: {rec.get('name', '')}
  Preview:
    {rec.get('preview', '')}
  Download URL: <url>{rec.get('url', '')}</url>
  CSV ID: {rec.get('csv_id') or 'not available'}
  Type: {rec.get('type', '')}
"""
    return ctx


def _build_export_payload() -> Dict[str, Any]:
    status = st.session_state.get("last_status") or "idle"
    messages = st.session_state.get("messages") or []
    if status == "idle" and messages:
        status = "ok"
    payload: Dict[str, Any] = {
        "ID": st.session_state.get("session_id", "streamlit_session"),
        "Context/Background": _context_for_export(),
        "Prompt": _latest_user_prompt(),
        "mode": st.session_state.get("mode", "Discovera"),
        "llm_provider": _current_llm_provider(),
        "llm_model": _active_model_name(),
        "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(_session_elapsed_seconds(), 2),
        "status": status,
        "error": st.session_state.get("last_error"),
        "output_text": _latest_output_text(),
        "tool_calls_json": st.session_state.get("tool_calls_log") or [],
        "tool_events_json": st.session_state.get("tool_events_log") or [],
        "uploaded_files": st.session_state.get("context_uploads") or [],
        "chat_messages": messages,
        "turn_runs": st.session_state.get("turn_runs") or [],
    }
    return payload


def _pdf_escape(text: str) -> str:
    safe = text.encode("latin-1", "replace").decode("latin-1")
    return safe.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _wrap_lines(text: str, width: int = 106) -> List[str]:
    out: List[str] = []
    for raw_line in text.splitlines() or [""]:
        if not raw_line:
            out.append("")
            continue
        wrapped = textwrap.wrap(
            raw_line,
            width=width,
            replace_whitespace=False,
            drop_whitespace=False,
            break_long_words=True,
            break_on_hyphens=False,
        )
        out.extend(wrapped or [""])
    return out


def _pdf_max_chars_for_width(width_pts: float, font_size: float = 10.0) -> int:
    avg_char_width = max(font_size * 0.52, 1.0)
    return max(14, int(width_pts / avg_char_width))


def _build_pdf_chat_blocks(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    summary = (
        f"Generated: {payload.get('run_date', '')}\n"
        f"Mode: {payload.get('mode', '')}\n"
        f"Provider/Model: {payload.get('llm_provider', '')} / {payload.get('llm_model', '')}\n"
        f"Status: {payload.get('status', '')}"
    )
    if payload.get("error"):
        summary += f"\nError: {_stringify(payload.get('error'), max_chars=500)}"
    blocks.append(
        {"kind": "meta", "label": "Discovera Session Export", "text": summary}
    )

    context_text = _stringify(payload.get("Context/Background", ""), max_chars=18000)
    if context_text.strip():
        blocks.append({"kind": "meta", "label": "Context", "text": context_text})

    for msg in payload.get("chat_messages") or []:
        role = str(msg.get("role") or "assistant").lower()
        is_foldable = bool(msg.get("foldable"))
        title = str(msg.get("title") or "")
        content_text = _stringify(msg.get("content", ""), max_chars=20000)

        if role == "user":
            kind = "user"
            label = "User"
        elif is_foldable:
            # Keep only explicit tool-call intent blocks from the transcript.
            # Skip tool result/event blocks to avoid duplicating turn_runs logs.
            is_tool_call_block = title.lower().startswith(
                "tool calls"
            ) or content_text.lower().startswith("tool calls:")
            if not is_tool_call_block:
                continue
            kind = "tool"
            label = title or "Tool calls"
        else:
            kind = "assistant"
            label = "Assistant"
        blocks.append(
            {
                "kind": kind,
                "label": str(label),
                "text": content_text,
            }
        )
    return blocks


def _build_chatbox_pdf(payload: Dict[str, Any]) -> bytes:
    blocks = _build_pdf_chat_blocks(payload)
    if not blocks:
        blocks = [
            {
                "kind": "meta",
                "label": "Discovera Session Export",
                "text": "(No content)",
            }
        ]

    page_width = 612.0
    page_height = 792.0
    margin = 34.0
    block_gap = 10.0
    pad_x = 10.0
    pad_top = 11.0
    pad_bottom = 10.0
    title_font = 9.0
    body_font = 9.5
    title_leading = 12.0
    body_leading = 12.0
    content_width = page_width - (2 * margin)

    def _style(kind: str) -> Dict[str, Any]:
        if kind == "user":
            return {"fill": (0.88, 0.94, 1.0), "align_right": True}
        if kind == "assistant":
            return {"fill": (0.95, 0.97, 0.99), "align_right": False}
        if kind == "tool":
            return {"fill": (0.93, 0.93, 0.93), "align_right": False}
        return {"fill": (0.94, 0.97, 0.94), "align_right": False}

    pages: List[List[str]] = [[]]
    y_top = page_height - margin

    for block in blocks:
        kind = str(block.get("kind") or "assistant")
        label = str(block.get("label") or "")
        text = str(block.get("text") or "")
        style = _style(kind)

        bubble_width = content_width * (0.78 if style["align_right"] else 0.9)
        bubble_x = margin + (
            content_width - bubble_width if style["align_right"] else 0.0
        )
        text_width = bubble_width - (2 * pad_x)

        max_chars = _pdf_max_chars_for_width(text_width, body_font)
        title_lines = _wrap_lines(label, width=max_chars)
        body_lines = _wrap_lines(text, width=max_chars)
        if not title_lines:
            title_lines = [""]
        if not body_lines:
            body_lines = [""]

        bubble_height = (
            pad_top
            + (len(title_lines) * title_leading)
            + 3.0
            + (len(body_lines) * body_leading)
            + pad_bottom
        )

        if y_top - bubble_height < margin:
            pages.append([])
            y_top = page_height - margin

        y_bottom = y_top - bubble_height
        r, g, b = style["fill"]
        cmds = pages[-1]

        cmds.extend(
            [
                "q",
                f"{r:.3f} {g:.3f} {b:.3f} rg",
                f"{bubble_x:.2f} {y_bottom:.2f} {bubble_width:.2f} {bubble_height:.2f} re f",
                "Q",
            ]
        )

        text_x = bubble_x + pad_x
        title_start_y = y_top - pad_top - title_font + 1.0
        body_start_y = (
            y_top - pad_top - (len(title_lines) * title_leading) - 3.0 - body_font + 1.0
        )

        cmds.extend(
            [
                "BT",
                f"/F2 {title_font:.2f} Tf",
                f"{title_leading:.2f} TL",
                f"1 0 0 1 {text_x:.2f} {title_start_y:.2f} Tm",
            ]
        )
        for line in title_lines:
            cmds.append(f"({_pdf_escape(line)}) Tj")
            cmds.append("T*")
        cmds.extend(
            [
                "ET",
                "BT",
                f"/F1 {body_font:.2f} Tf",
                f"{body_leading:.2f} TL",
                f"1 0 0 1 {text_x:.2f} {body_start_y:.2f} Tm",
            ]
        )
        for line in body_lines:
            cmds.append(f"({_pdf_escape(line)}) Tj")
            cmds.append("T*")
        cmds.append("ET")

        y_top = y_bottom - block_gap

    page_obj_ids: List[int] = []
    content_obj_ids: List[int] = []
    next_obj_id = 5
    for _ in pages:
        page_obj_ids.append(next_obj_id)
        content_obj_ids.append(next_obj_id + 1)
        next_obj_id += 2
    total_objects = next_obj_id - 1

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: Dict[int, int] = {}

    def _append_object(obj_id: int, body: bytes):
        offsets[obj_id] = len(out)
        out.extend(f"{obj_id} 0 obj\n".encode("latin-1"))
        out.extend(body)
        if not body.endswith(b"\n"):
            out.extend(b"\n")
        out.extend(b"endobj\n")

    kids = " ".join(f"{pid} 0 R" for pid in page_obj_ids)
    _append_object(1, b"<< /Type /Catalog /Pages 2 0 R >>\n")
    _append_object(
        2,
        f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>\n".encode("latin-1"),
    )
    _append_object(3, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\n")
    _append_object(4, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>\n")

    for page_obj_id, content_obj_id, commands in zip(
        page_obj_ids, content_obj_ids, pages
    ):
        stream = "\n".join(commands).encode("latin-1", "replace")
        content_body = (
            f"<< /Length {len(stream)} >>\nstream\n".encode("latin-1")
            + stream
            + b"\nendstream\n"
        )
        _append_object(content_obj_id, content_body)
        page_body = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_width:.0f} {page_height:.0f}] "
            f"/Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> "
            f"/Contents {content_obj_id} 0 R >>\n"
        ).encode("latin-1")
        _append_object(page_obj_id, page_body)

    xref_start = len(out)
    out.extend(f"xref\n0 {total_objects + 1}\n".encode("latin-1"))
    out.extend(b"0000000000 65535 f \n")
    for obj_id in range(1, total_objects + 1):
        offset = offsets.get(obj_id, 0)
        out.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))
    out.extend(
        (
            f"trailer\n<< /Size {total_objects + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_start}\n%%EOF\n"
        ).encode("latin-1")
    )
    return bytes(out)


def _build_pdf_export_bytes() -> bytes:
    payload = _build_export_payload()
    return _build_chatbox_pdf(payload)


def _build_json_export_text() -> str:
    return json.dumps(
        _build_export_payload(), indent=2, ensure_ascii=False, default=str
    )


def _export_file_stub() -> str:
    raw = st.session_state.get("session_id", "streamlit_session")
    safe = "".join(ch for ch in str(raw) if ch.isalnum() or ch in ("-", "_"))
    return safe or "streamlit_session"


def _reinit_session_for_mode(new_mode: str):
    # Reset all conversational/agent state and mode-dependent settings
    _reset_chat_state()
    st.session_state["session_id"] = datetime.now().strftime("streamlit_%Y%m%d_%H%M%S")
    st.session_state["mcp_connected"] = False
    st.session_state["mcp_auto_checked"] = False
    st.session_state["mcp_server"] = _default_addr_for_mode(new_mode)
    # Also reset context to avoid cross-mode leakage
    st.session_state["context"] = ""
    st.session_state["context_uploads"] = []
    st.session_state["context_upload_hashes"] = set()


def _ensure_session_defaults():
    if "mode" not in st.session_state:
        st.session_state["mode"] = "Discovera"
    if "prev_mode" not in st.session_state:
        st.session_state["prev_mode"] = st.session_state["mode"]
    if "mcp_server" not in st.session_state:
        st.session_state["mcp_server"] = _default_addr_for_mode(
            st.session_state["mode"]
        )
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = datetime.now().strftime(
            "streamlit_%Y%m%d_%H%M%S"
        )
    if "session_started_at" not in st.session_state:
        st.session_state["session_started_at"] = datetime.now().isoformat()
    if "checkpointer" not in st.session_state:
        st.session_state["checkpointer"] = InMemorySaver()

    defaults = {
        "messages": [],
        "tool_calls_log": [],
        "tool_events_log": [],
        "turn_runs": [],
        "last_error": None,
        "last_status": "idle",
        "llm_provider_selected": DEFAULT_LLM_PROVIDER,
        "openai_model": OPENAI_MODEL,
        "openai_api_key": "",
        "claude_model": CLAUDE_MODEL or "",
        "claude_api_key": "",
        "portkey_model": PORTKEY_MODEL,
        "portkey_api_key": "",
        "nvidia_model": NVIDIA_MODEL,
        "nvidia_api_key": "",
        "mcp_auto_checked": False,
        "mcp_connected": False,
        "context": "",
        "context_uploads": [],
        "context_upload_hashes": set(),
        "output_files": [],
        "output_file_urls": set(),
        "mcp_tool_infos": [],
        "mcp_tool_filter": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_ensure_session_defaults()


def _check_mcp_connection(addr: str, toast: bool = True) -> bool:
    url = httpx.URL(addr)
    ok = False
    notes: List[str] = []
    try:
        with httpx.Client(timeout=200.0, follow_redirects=True) as http:
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
        if ok:
            try:
                tools_loaded = _load_mcp_tools(addr)
                st.session_state["mcp_tool_infos"] = _serialize_tools(tools_loaded)
                if toast:
                    st.toast(f"Loaded {len(st.session_state['mcp_tool_infos'])} tools")
            except Exception as tool_exc:
                if toast:
                    st.toast(f"Tool load failed: {tool_exc}")
    except Exception as e:
        st.session_state["mcp_connected"] = False
        notes.append(str(e))

    if toast:
        status_msg = "ok" if ok else "failed"
        detail = " | ".join(notes) if notes else ""
        st.toast(f"MCP check {status_msg}. {detail}")
    return ok


with st.sidebar.expander("LLM Settings", expanded=False):
    st.selectbox(
        "Provider",
        list(SUPPORTED_LLM_PROVIDERS),
        key="llm_provider_selected",
    )
    selected_provider = _current_llm_provider()

    model_key = _model_state_key(selected_provider)
    api_key_key = _api_key_state_key(selected_provider)
    model_choice_key = f"{selected_provider}_model_choice"
    model_options = MODEL_PRESETS.get(selected_provider, [])
    current_model = (st.session_state.get(model_key) or "").strip()
    if not current_model:
        current_model = _active_model_name()
    default_choice = (
        current_model if current_model in model_options else MODEL_OTHER_OPTION
    )
    if model_choice_key not in st.session_state:
        st.session_state[model_choice_key] = default_choice
    elif st.session_state.get(model_choice_key) not in (
        model_options + [MODEL_OTHER_OPTION]
    ):
        st.session_state[model_choice_key] = MODEL_OTHER_OPTION

    selected_model_choice = st.selectbox(
        "Model",
        model_options + [MODEL_OTHER_OPTION],
        key=model_choice_key,
    )
    if selected_model_choice == MODEL_OTHER_OPTION:
        st.text_input("Custom Model", key=model_key)
    else:
        st.session_state[model_key] = selected_model_choice
        st.caption(f"Selected model: `{selected_model_choice}`")

    st.text_input("API Key", key=api_key_key, type="password")

with st.sidebar.expander("Mode", expanded=False):
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

if not st.session_state.get("mcp_auto_checked", False):
    auto_addr = st.session_state.get("mcp_server") or _default_addr_for_mode(
        st.session_state.get("mode", "Discovera")
    )
    _check_mcp_connection(auto_addr, toast=False)
    st.session_state["mcp_auto_checked"] = True

with st.sidebar.expander("Connection", expanded=False):
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
            _check_mcp_connection(addr, toast=True)
            st.session_state["mcp_auto_checked"] = True

    connected = bool(st.session_state.get("mcp_connected"))
    status_text = "🟢 connected" if connected else "🔴 disconnected"
    addr_text = st.session_state.get("mcp_server") or default_addr
    st.caption(f"Status: {status_text} · {addr_text}")


server_instructions = """
You are a biomedical research assistant. YOU MUST USE THE DISCOVERA MCP TOOLS to answer.
Be concise, results-style (e.g., “led to”, “resulted in”), and evidence-grounded.
Include inline PMIDs/PMCIDs and a references list. Elevate cross-modality/contrast overlaps;
mark single‑modality signals as secondary.

TRUST AND EVIDENCE RULES (STRICT)
- Do not infer or invent sample sizes, replicates, fold-changes, p-values, effect sizes, or metadata.
  Only state numbers explicitly present in the user context or tool outputs. If unknown, say “not specified”.
- In the Introduction/Context, restate only facts present in the provided context or file metadata.
  Do not add assumptions (e.g., number of replicates) if they are not given.

GRANULARITY ROUTING
- If the user asks a gene/pathway‑specific question: closely inspect enrichment result CSVs and tables; search
  for the query’s keywords; and perform deep follow‑up using Steps 4–6 to add mechanism and literature.
- If the user asks a broad/open question: emphasize global patterns (pathway ratios, dominant modules, cross‑contrast
  overlaps) and avoid over‑specific claims not supported by the inputs.

VERIFICATION GATE (BEFORE FINAL OUTPUT)
- Self‑check all numeric/specific claims. For each, identify its source (context text, CSV column, or tool step).
  Remove or rephrase any claim lacking a clear source to “not specified”.

TOOL USAGE RULES
- You MUST use tools to answer - do NOT skip enrichment or analysis steps.
- Follow the pipeline sequentially: intake → inspect → shape → enrich → analyze → literature.
- LIMIT TOOL CALLS PER TURN:
  - Maximum 3 tool calls per assistant turn for literature/query tools (query_string_rumma,
    query_table_rumma, enrich_rumma, web_search).
  - Maximum 1 enrichment call per turn (gsea_pipe, ora_pipe, run_deseq2_*).
  - Batch related queries into a SINGLE call when possible (e.g., one query_genes call with
    multiple genes instead of separate calls).
- AVOID REDUNDANT CALLS: Do not call the same tool multiple times with similar queries.
  Consolidate related lookups.
- If you need more information, spread calls across multiple turns rather than overwhelming
  a single turn.

# PIPELINED CALLS (one tool per turn; skip unnecessary steps)

0) Keyword extraction and Gene Prioritization
- Derive a compact list of salient, domain-relevant keywords from the Context and User Prompt.
  Focus on: gene/protein symbols, post‑translational modifications, disease/phenotype terms, tissues/compartments, pathway/module names,
  and treatment/mutation terms. Keep this list short (≈3–5 items) and use it to guide enrichment interpretation and filtering.
- Output (final visible result): 
  - Keywords: concise list of 3–5 domain keywords. 
  - Candidate Genes: short list (3–8) of plausible gene or protein symbols that are mechanistically tied to the extracted keywords.
  These will serve as the seed list for Steps 4-6.
0b) Keyword/key‑gene–driven dataset interrogation (REQUIRED)
- Before enrichment, actively analyze the input dataset(s) using the Step 0 keywords and candidate genes:
  case‑insensitive searches in gene/feature columns, quick per‑gene presence/absence
  and direction summaries where available,
  and focused subsetting of shaped tables. Use these findings to inform shaping thresholds and subsequent enrichment.

1) Intake
- If a CSV/URL is provided and no `csv_id` is already in context: call `csv_record` (copy URLs exactly) → file_id.
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
- Primary: Attempt enrichment using gsea_pipe(csv_id, hit_col, corr_col, gene_sets, threshold).
  If it fails due to missing or invalid ranking metrics (hit_col / corr_col), proceed to counts-based enrichment.
    - Fallback 1 (counts-based): If raw counts are available:
        - Build sample_groups from metadata when present; otherwise infer from column prefixes.
        - Run `run_deseq2_gsea_pipe(raw_counts_csv_id, sample_groups, gene_column, gene_sets, p_value_threshold, fdr_threshold)`.
        - Run `run_deseq2_ora_pipe(raw_counts_csv_id, sample_groups, gene_column, gene_sets, fdr_threshold)`.
    - Fallback 2 (ORA): If neither ranking metrics nor raw counts are available, use ora_pipe(csv_id, gene_col, gene_sets).
- **RUN BOTH WHEN AVAILABLE (RECOMMENDED)**
  - If a ranking metric exists AND a significant gene list can be defined (by FDR/p-value or top-N),
    run both methods: GSEA (directional, ranking-based) and ORA (counts-based) in separate turns.
  - For ORA, build an all-significant, direction-agnostic list to mirror author-style pathway mentions;
    optionally also test Up-only and Down-only lists when informative.
  - Synthesize findings across methods and label results by method.
  - The tool response shows only a preview by default: for GSEA, top 5 UP and top 5 DOWN per
    library; for ORA, the top ~20 pathways. These previews are not exhaustive.
  - **ALWAYS operate on the saved full results CSV returned by Step 4b**.

4b) Keyword‑guided filtering of enrichment outputs (REQUIRED)
- From the main GSEA/ORA results CSV produced in Step 4, create a focused subset using both the Step 0 keywords and candidate genes.
  Apply case‑insensitive `contains` filters on common term/name columns
  (e.g., term, name, pathway, description). Support optional negative filters when the prompt implies
  exclusions (not_contains).
- When multiple columns exist, OR-match across columns for inclusion and AND-apply any negatives.
- If the focused subset is empty, relax to stem/synonym variants; if still empty, clearly state
  “no keyword‑matching terms observed at the chosen threshold”.
- If the focused subset is empty OR the full results do not cover key genes/patterns implied by
  the user query and Step 0 keywords, RERUN enrichment with higher thresholds
  (increase p_value_threshold and/or fdr_threshold) and repeat
  the filtering on the new full results until the information is sufficient for the user query.
- Use `csv_filter`, `csv_select`, and related CSV tools to implement filtering reproducibly on the
  returned artifact (prefer tool‑based filtering over ad‑hoc parsing).
 
5) Leading edge and mechanism (REQUIRED)
- Extract leading-edge genes from enriched terms (up and down).
- Call `gene_info` with a batch of up to 10 genes (single call, not per-gene).
- Optionally call `sets_info_rumm` once for a key gene set.
- Optional: ONE `enrich_rumma` call on leading-edge genes to validate themes.
- Optional: Call `query_genes` to query the INDRA database for relationships among a list of genes.

6) Literature and prioritization (REQUIRED)
- Use `query_string_rumma` (free text) and/or `query_table_rumma` (table search) to tie terms to PMCs.
- Use `literature_trends` for time context.
- Run prioritize_genes to rank candidate genes within the disease/phenotype context (PubMed-grounded).
- Frame queries around observed biology (e.g., stress response, cell cycle, metabolism) 
  using the keywords and candidate genes from Step 0 to guide Steps 4–6.

7) Web search
- Use `web_search(query, num_results)` to find related literature and background information.
- Can be used for following up on literature and background information.


# REQUIRED OUTPUT
- Introduction/Context: mutation/treatment; tissue/cell line; assay(s)/contrast; dataset/paper
  reference. Briefly summarize definitive signals in the input. For any detail not present, state “not specified”.
- Keywords: list of keywords extracted from the Context and User Prompt.
- Candidate Genes: short list (3–8) of plausible gene or protein symbols that are mechanistically tied to the extracted keywords.
- Pathway Enrichments:
  - Report up/down ratios per library using `counts_by_database` (e.g., 62% up, 38% down, ratio 1.63).
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
    | ... | ... | ... | ... |... |

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
    if a user-mentioned item is absent in Step 4 outputs, state “not observed” with rationale.
  - **Try to include all shared genes between the up and down enriched pathways.**
- Comparisons: overlaps vs tissue‑specific; flag discordant single‑modality findings as lower confidence.
- Implications: relevance for therapy, resistance, or progression.
- Next Steps & Clarification: 2–4 concrete follow‑ups.
 - Explicit targeted check: state presence/absence (with rationale) of the key biological themes implied by
   the question and assay (e.g., replication/cell-cycle, stress responses, metabolism/mitochondria).
- References (only if validated in Steps 5-6): include only PMIDs/PMCIDs returned by those steps; omit
  the References section entirely if none are validated.

### CRITICAL: Citation Verification Protocol
**NEVER fabricate or hallucinate citations.** Follow this chain-of-thought verification:           
1. **Before finalizing any report**, perform a mental citation audit:
- List each citation you're about to include
- For each one, identify which tool call returned it
- If you cannot trace a citation to a tool result, **remove it**
           
2. **Format for verified citations**:
- Always indicate the source: "According to INDRA (PMID/PMCID: X)..." or "Literature search returned..."
- Include database attribution: "Retrieved from PubMed via literature_trends tool"
"""


baseline_server_instructions = """
You are an assistant MCP specializing in biomedical research, focusing on mutation effects, treatment responses,
pathway enrichments, and mechanistic biology. Your final report must be written in the style of a **scientific
Results section** — condensed, definitive, and directly grounded in the dataset and prompt being analyzed.

# RULES
- Please always look into the uploaded datasets and user prompt to make inferences.
  You cannot use any external knowledge.

# REQUIRED OUTPUT (ONLY INCLUDE THE SECTIONS THAT YOU HAVE ENOUGH INFORMATION TO FILL IN)
- Introduction/Context: mutation/treatment; tissue/cell line; assay(s)/contrast; dataset/paper
  reference. Briefly summarize definitive signals in the input. For any detail not present, state “not specified”.
- Keywords: list of keywords extracted from the Context and User Prompt.
- Candidate Genes: short list (3–8) of plausible gene or protein symbols that are mechanistically tied to the extracted keywords.
- Pathway Enrichments:
  - Report up/down ratios per library using `counts_by_database` (e.g., 62% up, 38% down, ratio 1.63).
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
    | ... | ... | ... | ... |... |

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
    if a user-mentioned item is absent in Step 4 outputs, state “not observed” with rationale.
  - **Try to include all shared genes between the up and down enriched pathways.**
- Comparisons: overlaps vs tissue‑specific; flag discordant single‑modality findings as lower confidence.
- Implications: relevance for therapy, resistance, or progression.
- Next Steps & Clarification: 2–4 concrete follow‑ups.
 - Explicit targeted check: state presence/absence (with rationale) of the key biological themes implied by
   the question and assay (e.g., replication/cell-cycle, stress responses, metabolism/mitochondria).
- References (only if validated in Steps 5-6): include only PMIDs/PMCIDs returned by those steps; omit
  the References section entirely if none are validated.

# CRITICAL GUARDRAILS
- Use exactly one tool per assistant turn.
- If input URLs are provided and no `csv_id` is already in context, call csv_record and copy the URL exactly as given.

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
        help="Files are uploaded to the local file server and summarized into the context.",
    )

    def _parse_tool_output(raw: Any) -> Dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, list):
            if len(raw) == 1 and isinstance(raw[0], str):
                raw = raw[0]
            else:
                return {"raw": raw}
        if isinstance(raw, str):
            txt = raw.strip()
            if txt.startswith("{") or txt.startswith("["):
                try:
                    parsed = json.loads(txt)
                    return parsed if isinstance(parsed, dict) else {"raw": parsed}
                except Exception:
                    return {"raw": raw}
            return {"raw": raw}
        return {"raw": str(raw)}

    def _auto_csv_record(
        csv_url: str,
        logical_name: str,
        csv_text: str | None = None,
    ) -> tuple[str | None, str | None]:
        mode = st.session_state.get("mode", "Discovera")
        server_url = st.session_state.get("mcp_server") or _default_addr_for_mode(mode)
        try:
            tools = _load_mcp_tools(server_url)
            tool = next(
                (t for t in tools if getattr(t, "name", "") == "csv_record"), None
            )
            if tool is None:
                return None, f"csv_record tool not found on MCP server: {server_url}"

            def _invoke(payload: Dict[str, Any]) -> tuple[str | None, str | None]:
                raw_result = asyncio.run(tool.ainvoke(payload))
                parsed = _parse_tool_output(raw_result)
                csv_id = parsed.get("id")
                if csv_id:
                    return str(csv_id), None
                err = parsed.get("error")
                if isinstance(err, dict):
                    return None, str(
                        err.get("message") or _stringify(err, max_chars=500)
                    )
                if err:
                    return None, str(err)
                return (
                    None,
                    f"Unexpected csv_record response: {_stringify(parsed, max_chars=500)}",
                )

            csv_id, err = _invoke({"name": logical_name, "csv_url": csv_url})
            if csv_id:
                return csv_id, None

            if csv_text:
                csv_id_text, err_text = _invoke(
                    {"name": logical_name, "csv_text": csv_text}
                )
                if csv_id_text:
                    return csv_id_text, None
                if err and err_text:
                    return None, f"{err}; fallback via csv_text failed: {err_text}"
                if err_text:
                    return None, err_text
            return None, err
        except Exception as exc:
            return None, str(exc)

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
        upload_stem = f"{sha[:12]}_{safe_name}"
        upload_filename = f"{upload_stem}{ext}"

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

        # Upload to local file server and derive a download URL.
        remote_url: str | None = None
        try:
            files = {
                "file": (
                    upload_filename,
                    raw,
                    "text/csv" if ext == ".csv" else "application/octet-stream",
                )
            }
            with httpx.Client(timeout=FILE_SERVER_TIMEOUT_SECONDS) as client:
                resp = client.post(f"{FILE_SERVER_BASE_URL}/upload", files=files)
                resp.raise_for_status()
                payload = resp.json() if resp.content else {}
            saved_name = (payload or {}).get("filename") or upload_filename
            remote_url = str(
                httpx.URL(f"{FILE_SERVER_BASE_URL}/download").copy_add_param(
                    "filename", saved_name
                )
            )
        except Exception:
            remote_url = None

        if not remote_url:
            return None

        csv_id = None
        csv_record_error = None
        if ext == ".csv":
            text_for_record = None
            try:
                text_for_record = raw.decode("utf-8-sig", errors="ignore")
            except Exception:
                text_for_record = None
            csv_id, csv_record_error = _auto_csv_record(
                csv_url=remote_url,
                logical_name=upload_stem,
                csv_text=text_for_record,
            )

        rec = {
            "hash": sha,
            "name": file_obj.name,
            "type": (ext[1:] if ext.startswith(".") else ext).lower(),
            "preview": preview,
            "url": remote_url,
            "csv_id": csv_id,
        }
        if csv_record_error:
            rec["csv_record_error"] = csv_record_error
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
                csv_id = rec.get("csv_id")
                if csv_id:
                    st.markdown(
                        f"- `{rec['name']}` → [download link]({rec['url']}) · `csv_id: {csv_id}`"
                    )
                else:
                    st.markdown(
                        f"- `{rec['name']}` → [download link]({rec['url']}) · `csv_id: not available`"
                    )
                    if rec.get("csv_record_error"):
                        st.caption(f"csv_record failed: {rec['csv_record_error']}")

# Create a placeholder so we can live-update Outputs during streaming
outputs_placeholder = st.sidebar.empty()


def render_outputs():
    with outputs_placeholder.container():
        with st.expander("Outputs", expanded=True):
            _outputs = st.session_state.get("output_files") or []
            if _outputs:
                for _rec in _outputs:
                    _name = _rec.get("name") or "download"
                    _url = _rec.get("url") or _rec.get("file_server_url") or ""
                    if _url:
                        st.markdown(f"- [{_name}]({_url})")
            else:
                st.caption("No outputs yet.")


# Initial render
render_outputs()

with st.sidebar.expander("MCP Tools", expanded=True):
    connected = bool(st.session_state.get("mcp_connected"))
    addr_text = st.session_state.get("mcp_server") or _default_addr_for_mode(
        st.session_state.get("mode", "Discovera")
    )
    cols = st.columns([3, 1])
    with cols[0]:
        st.caption(f"Server: {addr_text}")
    with cols[1]:
        if st.button("Refresh", use_container_width=True, key="btn_refresh_tools"):
            try:
                tools_loaded = _load_mcp_tools(addr_text)
                st.session_state["mcp_tool_infos"] = _serialize_tools(tools_loaded)
                st.toast(f"Loaded {len(st.session_state['mcp_tool_infos'])} tools")
            except Exception as tool_exc:
                st.toast(f"Tool load failed: {tool_exc}")

    if not connected:
        st.caption("Not connected.")
    else:
        tool_infos = st.session_state.get("mcp_tool_infos") or []
        filter_text = st.text_input(
            "Filter tools",
            key="mcp_tool_filter",
            placeholder="Filter by name or description",
        )
        ft = (filter_text or "").strip().lower()
        if ft:
            filtered = [
                info
                for info in tool_infos
                if ft in info.get("name", "").lower()
                or ft in (info.get("description", "") or "").lower()
            ]
        else:
            filtered = tool_infos
        st.caption(f"Showing {len(filtered)} of {len(tool_infos)} tools")
        if not filtered:
            st.caption("No tools to display. Click Refresh to load tools.")
        for info in filtered:
            name = info.get("name") or "tool"
            desc = info.get("description") or ""
            schema_json = info.get("schema_json")
            st.markdown(f"**{name}**")
            if desc:
                st.caption(desc)
            if schema_json:
                with st.expander("Input schema", expanded=False):
                    st.code(schema_json, language="json")

status_cols = st.columns(3)
with status_cols[0]:
    if st.button("Clear Chat", use_container_width=True):
        _reset_chat_state()
        st.session_state["session_id"] = datetime.now().strftime(
            "streamlit_%Y%m%d_%H%M%S"
        )
        st.rerun()

has_export_content = bool(
    (st.session_state.get("messages") or [])
    or (st.session_state.get("tool_calls_log") or [])
    or (st.session_state.get("tool_events_log") or [])
)
json_export_text = _build_json_export_text() if has_export_content else "{}"
pdf_export_bytes = _build_pdf_export_bytes() if has_export_content else b""
file_stub = _export_file_stub()

with status_cols[1]:
    st.download_button(
        "Export JSON",
        data=json_export_text,
        file_name=f"{file_stub}.json",
        mime="application/json",
        use_container_width=True,
        disabled=not has_export_content,
        key="btn_export_json",
    )

with status_cols[2]:
    st.download_button(
        "Export PDF",
        data=pdf_export_bytes,
        file_name=f"{file_stub}.pdf",
        mime="application/pdf",
        use_container_width=True,
        disabled=not has_export_content,
        key="btn_export_pdf",
    )

st.divider()

# --- Chat state ---
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
            st.markdown(content)

# Compose
prompt = st.chat_input("Ask anything…")


def _resolved_api_key(provider: str) -> str:
    session_key_name = _api_key_state_key(provider)
    env_fallback = {
        "openai": OPENAI_API_KEY,
        "claude": CLAUDE_API_KEY,
        "portkey": PORTKEY_API_KEY,
        "nvidia": NVIDIA_API_KEY,
    }.get(provider)
    return (st.session_state.get(session_key_name) or env_fallback or "").strip()


def _build_model():
    provider = _current_llm_provider()
    api_key = _resolved_api_key(provider)
    if not api_key:
        raise ValueError(
            f"{provider.capitalize()} API key is required in sidebar LLM settings."
        )

    if provider == "portkey":
        portkey_headers = createHeaders(
            api_key=api_key,
            metadata={"_user": "yfw215"},
        )
        return ChatOpenAI(
            model=(st.session_state.get("portkey_model") or PORTKEY_MODEL),
            max_tokens=65536,
            base_url=PORTKEY_BASE_URL,
            default_headers=portkey_headers,
        )
    if provider == "claude":
        model_name = (
            st.session_state.get("claude_model") or CLAUDE_MODEL or ""
        ).strip()
        if not model_name:
            raise ValueError("Claude model is required in sidebar LLM settings.")
        return ChatAnthropic(
            model=model_name,
            temperature=0,
            api_key=api_key,
        )
    if provider == "nvidia":
        return ChatOpenAI(
            model=(st.session_state.get("nvidia_model") or NVIDIA_MODEL),
            max_tokens=100000,
            api_key=api_key,
            base_url=NVIDIA_BASE_URL,
            extra_body={"chat_template_kwargs": {"thinking": True}},
        )
    return ChatOpenAI(
        model=(st.session_state.get("openai_model") or OPENAI_MODEL),
        reasoning_effort="high",
        max_tokens=100000,
        api_key=api_key,
    )


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
        ctx += "Uploaded files (stored in local file server):\n"
        for rec in uploads:
            ctx += f"""
Name: {rec['name']}
  Preview:
    {rec['preview']}
  Download URL: <url>{rec['url']}</url>
  CSV ID: {rec.get('csv_id') or 'not available'}
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
                url = v.get("file_server_url")
                if url:
                    seen = st.session_state.get("output_file_urls") or set()
                    if url not in seen:
                        rec = {
                            "name": v.get("name") or "download",
                            "url": url,
                            "size_bytes": v.get("size_bytes"),
                            "created_at": v.get("created_at"),
                            "category": v.get("category"),
                            "origin_tool": v.get("origin_tool"),
                            "file_server_url": v.get("file_server_url"),
                        }
                        out_list = st.session_state.get("output_files") or []
                        out_list.append(rec)
                        st.session_state["output_files"] = out_list
                        seen.add(url)
                        st.session_state["output_file_urls"] = seen
                        render_outputs()


def stream_langgraph(prompt: str):
    run_started_at = datetime.now()
    run_tool_calls: List[Dict[str, Any]] = []
    run_tool_events: List[Dict[str, Any]] = []
    run_text_chunks: List[str] = []

    mode = st.session_state.get("mode", "Discovera")
    default_addr = _default_addr_for_mode(mode)
    server_url = st.session_state.get("mcp_server") or default_addr

    def _run_summary(status: str, error: str | None = None) -> Dict[str, Any]:
        return {
            "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": round(
                (datetime.now() - run_started_at).total_seconds(), 2
            ),
            "status": status,
            "error": error,
            "output_text": "".join(run_text_chunks),
            "tool_calls": run_tool_calls,
            "tool_events": run_tool_events,
        }

    # Build tools and agent
    try:
        tools = _load_mcp_tools(server_url) + [web_search]
    except Exception as e:
        error_text = f"[MCP tools error] {e}"
        yield ("error", error_text)
        yield ("run_summary", _run_summary(status="error", error=error_text))
        return

    try:
        model = _build_model()
        agent = create_react_agent(
            model,
            tools=tools,
            prompt=prompt,
            version="v2",
            checkpointer=st.session_state["checkpointer"],
        )
    except Exception as e:
        error_text = f"[Agent init error] {e}"
        yield ("error", error_text)
        yield ("run_summary", _run_summary(status="error", error=error_text))
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
            for _, node_update in update.items():
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
                            run_text_chunks.append(content_text)
                            yield ("delta", content_text)
                        # Surface tool call intents
                        tool_calls = getattr(msg, "tool_calls", None) or []
                        if tool_calls:
                            calls_render = []
                            for call in tool_calls:
                                name = call.get("name")
                                args = call.get("args")
                                run_tool_calls.append({"name": name, "args": args})
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
                        tool_payload = (
                            parsed_json if parsed_json is not None else tool_text
                        )
                        run_tool_events.append(
                            {
                                "tool": tool_name,
                                "payload": tool_payload,
                                "is_error": is_error,
                            }
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
        yield ("run_summary", _run_summary(status="ok"))
    except Exception as e:
        error_text = f"[Run error] {e}"
        yield ("error", error_text)
        yield ("run_summary", _run_summary(status="error", error=error_text))


if prompt:
    st.session_state["last_status"] = "running"
    st.session_state["last_error"] = None

    # 1) append user to UI transcript and show it
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2) stream from LangGraph agent
    # 2) stream from LangGraph agent, rendering events in time order (no placeholder)
    summary_seen = False
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
        elif kind == "run_summary" and isinstance(payload, dict):
            summary_seen = True
            run_record = dict(payload)
            run_record["prompt"] = prompt
            run_record["mode"] = st.session_state.get("mode", "Discovera")
            run_record["llm_provider"] = _current_llm_provider()
            run_record["llm_model"] = _active_model_name()
            runs = st.session_state.get("turn_runs") or []
            runs.append(run_record)
            st.session_state["turn_runs"] = runs

            calls = payload.get("tool_calls") or []
            events = payload.get("tool_events") or []
            st.session_state["tool_calls_log"] = (
                st.session_state.get("tool_calls_log") or []
            ) + calls
            st.session_state["tool_events_log"] = (
                st.session_state.get("tool_events_log") or []
            ) + events
            st.session_state["last_status"] = payload.get("status") or "ok"
            st.session_state["last_error"] = payload.get("error")

    if not summary_seen and st.session_state.get("last_status") == "running":
        st.session_state["last_status"] = "ok"

    # Refresh the top controls so export buttons are enabled immediately
    # after a completed run (same interaction cycle).
    st.rerun()
