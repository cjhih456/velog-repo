# 01c_convert_hermes_traces.py
"""lambda/hermes-agent-reasoning-traces (ShareGPT) -> 기존 파이프라인의
messages/tools(OpenAI-ish) 포맷으로 변환. Base LoRA 학습용 + 카테고리별로
fullstack/plan/qa 데이터에 보충 투입할 수 있도록 category도 함께 저장.

두 가지를 반드시 지킨다 (02_train_lora.py에서 나던 에러들의 재발 방지):
  1) tools는 리스트가 아니라 "JSON 문자열"로 저장한다. 도구마다 parameters
     JSON-schema 구조가 제각각이라, HF datasets가 JSONL을 Arrow로 로드할 때
     리스트-of-dict 컬럼의 스키마를 잘못 추론하거나 필드를 유실시킬 수 있다
     (ValueError: Tools should either be a JSON schema... 의 유력 원인).
     문자열로 저장해두면 스키마 추론 문제 자체가 발생하지 않고, 학습 스크립트에서
     json.loads로 그대로 복원해 쓰면 된다.
  2) 실제 저장 전에 tokenizer.apply_chat_template로 "진짜로 렌더링되는지"
     검증하고, 실패하는 샘플은 걸러낸다. 문제 있는 데이터가 애초에 학습 스크립트로
     넘어가지 않게 해서 훈련 도중 크래시를 방지한다.
"""
import json, argparse, random
from pathlib import Path
from datasets import load_dataset

ROLE_MAP = {"system": "system", "human": "user", "gpt": "assistant", "tool": "tool"}

# 사용자의 5개 태스크에 보충 투입 가능한 카테고리 (참고용, quote/design은 대응 없음)
SUPPLEMENT_MAP = {
    "fullstack": {"Terminal & Coding", "Repository Tasks"},
    "plan": {"Planning & Organization"},
    # qa는 카테고리가 아닌 subcategory 레벨 필터가 필요 -> 아래 --qa-subcategory-keywords 참고
}


def convert_conversation(conversations):
    messages = []
    for turn in conversations:
        role = ROLE_MAP.get(turn["from"])
        if role is None:
            continue  # 알 수 없는 role은 건너뛰고 콘솔에 남기는 편이 안전
        # <think>/<tool_call>/<tool_response> 태그는 원문 그대로 content에 유지한다.
        # (Qwen3.6 chat template이 assistant/tool의 plain-string content는
        #  내부 태그를 건드리지 않고 <|im_start|>...<|im_end|>로만 감싸므로 안전하지만,
        #  실제 렌더링 결과는 아래 validate_example()로 반드시 검증한다.)
        messages.append({"role": role, "content": turn["value"]})
    return messages


def normalize_tools(raw_tools_field):
    """리포마다 tools 필드가 JSON 문자열이거나 이미 파싱된 리스트일 수 있어 둘 다 받는다.
    항상 파이썬 list[dict]로 정규화해서 반환 (아직 문자열로 재직렬화하지 않은 상태)."""
    if not raw_tools_field:
        return []
    if isinstance(raw_tools_field, str):
        try:
            return json.loads(raw_tools_field)
        except json.JSONDecodeError:
            return []
    if isinstance(raw_tools_field, list):
        return raw_tools_field
    return []


def validate_example(tokenizer, messages, tools):
    """실제 apply_chat_template 렌더링이 성공하는지 검증. 실패 시 (False, 에러메시지)."""
    try:
        rendered = tokenizer.apply_chat_template(
            messages, tools=tools or None, tokenize=False, add_generation_prompt=False
        )
        if not isinstance(rendered, str) or not rendered.strip():
            return False, "empty rendering"
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--configs", nargs="+", default=["kimi", "glm-5.1"])
    ap.add_argument(
        "--kimi-weight",
        type=int,
        default=2,
        help="kimi 추론 깊이가 더 깊으므로 base 학습셋에서 반복 사용할 배수",
    )
    ap.add_argument(
        "--val-ratio",
        type=float,
        default=0.05,
        help="base_val.jsonl 비율. weight 적용 전(중복 전) 원본 기준으로 분리한다",
    )
    ap.add_argument(
        "--qa-subcategory-keywords",
        nargs="+",
        default=["test", "pytest", "unit test", "code review", "review", "qa"],
    )
    ap.add_argument(
        "--base-model",
        default="Qwen/Qwen3.6-27B",
        help="렌더링 검증에 쓸 토크나이저(모델 가중치는 로드하지 않음)",
    )
    ap.add_argument(
        "--skip-validate",
        action="store_true",
        help="검증을 건너뛰고 빠르게 변환만 하고 싶을 때 (권장하지 않음)",
    )
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    tokenizer = None
    if not args.skip_validate:
        # 토크나이저만 로드 (프로세서 전체/모델 가중치는 불필요, 검증용이므로 가볍게)
        from transformers import Qwen3VLProcessor

        tokenizer = Qwen3VLProcessor.from_pretrained(
            args.base_model, trust_remote_code=True
        ).tokenizer

    base_pool = []  # (example, weight) — weight 중복은 train/val 분리 이후에 적용
    supplement = {k: [] for k in SUPPLEMENT_MAP}
    qa_rows = []
    n_seen, n_dropped = 0, 0
    drop_reasons = {}

    for cfg in args.configs:
        ds = load_dataset("lambda/hermes-agent-reasoning-traces", cfg, split="train")
        weight = args.kimi_weight if cfg == "kimi" else 1
        for row in ds:
            n_seen += 1
            messages = convert_conversation(row["conversations"])
            tools = normalize_tools(row.get("tools"))

            if tokenizer is not None:
                ok, reason = validate_example(tokenizer, messages, tools)
                if not ok:
                    n_dropped += 1
                    drop_reasons[reason] = drop_reasons.get(reason, 0) + 1
                    continue  # 렌더링 실패 샘플은 애초에 저장하지 않는다

            # tools는 항상 JSON 문자열로 저장 (datasets의 Arrow 스키마 추론 이슈 회피)
            example = {
                "messages": messages,
                "tools": json.dumps(tools, ensure_ascii=False),
                "category": row["category"],
                "subcategory": row["subcategory"],
            }

            # 전체 카테고리를 base 학습셋 후보로 포함 (범용 에이전트 행동 학습이 목적)
            base_pool.append((example, weight))

            for task, cats in SUPPLEMENT_MAP.items():
                if row["category"] in cats:
                    supplement[task].append(example)

            if any(
                kw in row["subcategory"].lower() for kw in args.qa_subcategory_keywords
            ):
                qa_rows.append(example)

    if tokenizer is not None:
        print(
            f"\n검증: {n_seen}건 중 {n_dropped}건 제외 ({n_dropped/max(n_seen,1)*100:.1f}%)"
        )
        for reason, cnt in sorted(drop_reasons.items(), key=lambda x: -x[1])[:10]:
            print(f"  {cnt:5d}건 - {reason[:150]}")

    # --- base_train / base_val 분리 ---
    # 반드시 weight 중복 적용 "이전"의 고유 샘플 단위로 분리한다.
    # 먼저 중복시키고 나서 섞어서 자르면 같은 샘플이 train과 val에 동시에 들어가는
    # 데이터 누수(leakage)가 생겨 검증 지표를 왜곡시키기 때문이다.
    rng = random.Random(42)
    rng.shuffle(base_pool)
    n_val = max(1, int(len(base_pool) * args.val_ratio))
    val_pool, train_pool = base_pool[:n_val], base_pool[n_val:]

    # val은 원본 분포 그대로 평가하기 위해 weight 중복 없이 1건씩만 사용한다.
    base_val_rows = [ex for ex, _ in val_pool]
    # train만 kimi-weight를 반영해 반복 사용한다.
    base_train_rows = [ex for ex, w in train_pool for _ in range(w)]
    rng.shuffle(base_train_rows)

    def dump(rows, path):
        with open(path, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{path}: {len(rows)}건")

    dump(base_train_rows, out_root / "base_train.jsonl")
    dump(base_val_rows, out_root / "base_val.jsonl")
    for task, rows in supplement.items():
        dump(rows, out_root / f"{task}_hermes_supplement.jsonl")
    dump(qa_rows, out_root / "qa_hermes_supplement.jsonl")

    print(
        f"\nbase pool 원본(중복 전) 총 {len(base_pool)}건 -> train {len(train_pool)} / val {len(val_pool)}"
        f" (val-ratio={args.val_ratio})"
    )
    print("주의: quote/design 태스크에 대응하는 카테고리가 없습니다.")
    print(
        "      base_train.jsonl/base_val.jsonl만 Base LoRA 학습에 쓰고, *_hermes_supplement.jsonl은"
    )
    print(
        "      기존 01_prepare_datasets.py 산출물(<task>_train.jsonl)에 append해서 보강용으로만 쓰세요."
    )


if __name__ == "__main__":
    main()
