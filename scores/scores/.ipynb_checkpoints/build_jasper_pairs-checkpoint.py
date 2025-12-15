import json
from pathlib import Path

INPUT_PATH = Path("synthetic/conversations/conversations.json")
OUTPUT_PATH = Path("jasper_train_pairs.jsonl")

def iter_conversations(path: Path):
    """假设文件是一个 JSON 数组：[{...}, {...}, ...]"""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    for conv in data:
        yield conv

def main():
    n_pairs = 0
    with OUTPUT_PATH.open("w", encoding="utf-8") as out_f:
        for conv in iter_conversations(INPUT_PATH):
            messages = conv.get("messages", [])
            last_user_text = None

            for msg in messages:
                speaker = msg.get("speaker")
                text = (msg.get("text") or "").strip()
                if not text:
                    continue

                if speaker == "user":
                    # 当前轮的检索 query
                    last_user_text = text

                elif speaker == "agent" and last_user_text:
                    # 该轮的答案 + 检索到的 contexts
                    contexts = msg.get("contexts") or []

                    # 1) user 问题 <-> 每个检索文本
                    for ctx in contexts:
                        ctx_text = (ctx.get("text") or "").strip()
                        if not ctx_text:
                            continue
                        record = {
                            "query": last_user_text,
                            "passage": ctx_text,
                            "type": "query_ctx"
                        }
                        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        n_pairs += 1

                    # 2) 可选：user 问题 <-> agent 回答（也可注释掉）
                    if text:
                        record = {
                            "query": last_user_text,
                            "passage": text,
                            "type": "query_answer"
                        }
                        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        n_pairs += 1

    print(f"Saved {n_pairs} training pairs to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
