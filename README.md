## Discovera Streamlit Chat UI

A lightweight Streamlit chat client for Discovera’s FastMCP agent. It lets you connect to a remote MCP server, add study context (and optional CSV/TXT uploads), and interact with a LangGraph-powered agent that performs enrichment, literature search, and gene-centric analyses.


### Key Features

- **MCP connection panel**: Enter the SSE endpoint and verify connectivity.
- **Context composer**: Add free-form study context; optionally upload CSV/TXT files to a file server with download links.
- **Streaming responses**: Agent thoughts and tool events render in real time.
- **Tool event panels**: Foldable sections show tool call intents and results.
- **Copy view**: Toggle assistant messages between rich text and a copy-friendly view.


## Prerequisites

- Python 3.8+
- An OpenAI API key
- A reachable MCP server SSE endpoint (e.g., FastMCP)
- A reachable file server endpoint (required only for file uploads)


## Installation

From the repository root:

```bash
pip install -r src/discovera_streamlit/requirements.txt
```

Or inside this folder:

```bash
cd src/discovera_streamlit
pip install -r requirements.txt
```


## Configuration

Create a `.env` file alongside `app.py` (or export these variables in your shell):

```
# OpenAI
OPENAI_API_KEY=sk-proj-...
# Optional: defaults to gpt-4o-mini in code
OPENAI_MODEL=gpt-5-nano

# MCP Server (SSE endpoint)
MCP_SERVER_ADDRESS=https://discovera-fastmcp.users.hsrn.nyu.edu/sse
# For local/dev you can use an ngrok tunnel
# MCP_SERVER_ADDRESS=https://<your-ngrok-subdomain>.ngrok-free.app/sse

# Timeouts (optional)
MCP_TIMEOUT_MS=3500

# File server (required for file uploads via the sidebar and batch scripts)
FILE_SERVER_BASE_URL=http://127.0.0.1:8001
FILE_SERVER_TIMEOUT_SECONDS=30
```

Notes:
- File uploads are optional. Without a reachable `FILE_SERVER_BASE_URL`, the chat still works but uploads are disabled.


## Run

From the repository root:

```bash
streamlit run src/discovera_streamlit/app.py
```

Or inside this folder:

```bash
cd src/discovera_streamlit
streamlit run app.py
```


## How to Use

1. **Check MCP connection**
   - Open the sidebar → Connection → verify or edit the MCP Server URL → click “Check MCP”.
   - You should see a toast and a status indicator (🟢/🔴).

2. **Add study context**
   - In the sidebar → Context, enter background, goals, datasets, and constraints.

3. **(Optional) Upload CSV/TXT files**
   - Drop CSV/TXT files in the uploader to include them in context.
   - Files are uploaded to the file server and shown with download links; the agent receives those links in the prompt and can register/read them via csv tools.

4. **Chat with the agent**
   - Use the chat input at the bottom. The agent streams intermediate thoughts, tool calls, and results.
   - Tool calls and outputs appear as foldable panels. Click to expand for details.
   - Use “Copy view” on assistant messages if you need a plain-text snapshot.

5. **Iterate**
   - Refine your question, add more context, or upload additional files. The agent keeps an in-memory thread for the session.


## Batch Processing

The repository includes two batch runner scripts for processing multiple tasks programmatically:

### `batch_runner_keypoints.py` - Breakdown Format

For CSV files where Task ID and Context are merged vertically across multiple keypoints. Each keypoint gets its own query.

**CSV Format:**
- Multiple rows can share the same Task ID and Context (merged vertically)
- Each row has a unique Keypoints identifier and Fine-grained Prompt
- Example: `Benchmark - verified - Breakdowns.csv`

**Usage:**
```bash
python scripts/batch_runner_keypoints.py \
  --input "Benchmark - verified - Breakdowns.csv" \
  --mode Discovera \
  --datasets-dir datasets \
  --task-ids 1,2.1,2.3,3.1,3.2,4,5.1,5.2,6,7.1,7.2,8,9.1,9.2,9.3,9.4,9.5,10,11.1,11.2,11.3,12,13,14.1,14.2,14.3,15 \
  --llm-provider openai
```

**Default Column Names:**
- `--task-id-col`: "Task ID"
- `--context-col`: "Context"
- `--prompt-col`: "Fine-grained Prompt"
- `--keypoints-col`: "Keypoints"

**Output:** Files are saved as `outputs/{mode}/task_{task_id}_{keypoints}.json`

### `batch_runner_single.py` - Single Prompt Format

For CSV files with one prompt per task. Each row represents a complete task.

**CSV Format:**
- Each row has a unique Task ID, Context, and Prompt
- Example: `Benchmark - verified - Benchmark.csv`

**Usage:**
```bash
python scripts/batch_runner_single.py \
  --input "Benchmark - verified - Benchmark.csv" \
  --mode Discovera \
  --datasets-dir datasets \
  --task-ids 3.2 \
  --llm-provider nvidia --output ./output_qwen3_235b_0120
```

**Default Column Names:**
- `--task-id-col`: "ID"
- `--context-col`: "Context/Background"
- `--prompt-col`: "Prompt"

**Output:** Files are saved as `outputs/{mode}/task_{task_id}.json`

### Common Arguments

Both scripts support the following arguments:

- `--input` (required): Path to input CSV/TSV/Parquet file
- `--output`: Path to output directory (defaults to `./outputs`)
- `--datasets-dir`: Path to datasets directory (defaults to `./datasets`)
- `--mode`: Agent mode - `Discovera` or `Baseline` (default: `Discovera`)
- `--task-ids`: Comma-separated list of task IDs to run (optional, processes all if omitted)
- `--server-url`: Override MCP server URL (defaults to env or mode-specific default)
- `--llm-provider`: LLM provider - `openai`, `claude`, or `portkey` (defaults to `LLM_PROVIDER` env var, or `openai`)

### Dataset Organization

The batch runners look for datasets in `{datasets-dir}/{task_id}/` directories. CSV files found in these directories are automatically uploaded to the file server and included in the context.
Uploaded CSVs are also registered via `csv_record` by default, and resulting `csv_id` values are included in context and output JSON.

**Example directory structure:**
```
datasets/
  ├── 3.2/
  │   ├── raw_counts.csv
  │   └── metadata.csv
  └── 4/
      └── data.csv
```

### Output Format

Each task produces a JSON file containing:
- Task metadata (ID, context, prompt)
- Agent output text
- Tool calls and events
- Uploaded file records
- Status and error information (if any)


## Tips and Troubleshooting

- **OPENAI_API_KEY missing**: The app will show an error and stop. Set it in `.env` or your shell and restart.
- **MCP connection fails**: Double-check the full SSE endpoint (often ends with `/sse`). Ensure the server is reachable and not blocked by firewalls.
- **Uploads not working**: Verify `FILE_SERVER_BASE_URL` points to a running file server and that `/upload` and `/download` endpoints are reachable.
- **Large files**: Prefer CSV. The app creates short previews and file-server download links.


## Advanced Configuration

- The system instructions sent to the agent are defined in `server_instructions` within `app.py`.
- The app uses LangGraph’s `create_react_agent` and an in-memory checkpointer. To change thread identifiers or agent behavior, edit `LANGGRAPH_CONFIG` and agent initialization in `app.py`.
