# 01d_spotcheck_template.py
import json, argparse
from transformers import Qwen3VLProcessor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--file", required=True)
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    tok = Qwen3VLProcessor.from_pretrained(args.model, trust_remote_code=True).tokenizer
    with open(args.file, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= args.n:
                break
            row = json.loads(line)
            # tools는 01c_convert_hermes_traces.py에서 JSON 문자열로 저장되므로 복원한다.
            tools_field = row.get("tools")
            tools = (
                json.loads(tools_field) if isinstance(tools_field, str) else tools_field
            )
            rendered = tok.apply_chat_template(
                row["messages"],
                tools=tools or None,
                tokenize=False,
                add_generation_prompt=False,
            )
            print(f"=== sample {i} (category={row.get('category')}) ===")
            print(rendered[:2000])
            print("...\n")


if __name__ == "__main__":
    main()
