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
  --task-ids 3.2 \
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
  --llm-provider openai
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
