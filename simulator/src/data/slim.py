"""Pass 1 (§2.3): stream the 6.4GB conversations table into a slim parquet with only the rows/columns turn building needs."""
import pyarrow as pa, pyarrow.parquet as pq
from pathlib import Path

D = Path(__file__).resolve().parents[2] / "data"
KEEP = {"user_prompt", "assistant_response", "assistant_thinking", "tool_use", "tool_result", "queue_operation"}
COLS = ["session_id", "turn_number", "role", "turn_type", "content", "queue_op_subtype", "tool_name",
        "file_path", "command", "tool_input_json", "timestamp", "prompt_intent", "prompt_pushback", "is_conversational"]
CAP = {"user_prompt": 8000, "assistant_response": 4000, "assistant_thinking": 300, "tool_result": 400, "tool_use": 0, "queue_operation": 600}

src = pq.ParquetFile(D / "raw/SWE-chat/conversations.parquet")
(D / "processed").mkdir(exist_ok=True)
w = None
for i in range(src.num_row_groups):
    df = src.read_row_group(i, columns=COLS).to_pandas()
    df = df[df.turn_type.isin(KEEP)].copy()
    df["input_chars"] = df.tool_input_json.str.len().fillna(0).astype("int32")
    df["content_chars"] = df.content.str.len().fillna(0).astype("int32")
    df["content"] = [None if c is None else c[:CAP[t]] for c, t in zip(df.content, df.turn_type)]
    df = df.drop(columns=["tool_input_json"])
    t = pa.Table.from_pandas(df, preserve_index=False)
    if w is None:  # first group can have all-null columns, so pin the schema from the source
        sch = pa.schema([src.schema_arrow.field(c) for c in COLS if c != "tool_input_json"] +
                        [pa.field("input_chars", pa.int32()), pa.field("content_chars", pa.int32())])
        w = pq.ParquetWriter(D / "processed/conv_slim.parquet", sch)
    t = t.select(w.schema.names).cast(w.schema)
    w.write_table(t)
    if i % 100 == 0: print(i, src.num_row_groups, flush=True)
w.close()
