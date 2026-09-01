# 02_train_lora.py (Unsloth 버전)
"""Unsloth의 FastVisionModel로 Qwen3.6-27B(Qwen3.5 계열, 비전 인코더 포함 Causal LM)를
로드해 QLoRA 학습 속도를 높인다. FastLanguageModel이 아닌 FastVisionModel을 쓰는 이유는
Qwen3.5/3.6이 "비전 인코더가 달린 통합 VLM"으로 등록되어 있기 때문 (Unsloth 공식 Qwen3.5
가이드 기준). unsloth는 반드시 다른 어떤 transformers/trl import보다도 먼저 import해야
내부 패치가 정상 적용된다."""
import argparse, platform, json

# --- unsloth는 항상 최상단에서 가장 먼저 import (내부 monkey-patch 순서 요구사항) ---
import unsloth
from unsloth import FastModel, is_bfloat16_supported

import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen3.6-27B")
    ap.add_argument(
        "--task",
        required=True,
        choices=["base"],
    )
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument(
        "--finetune-vision",
        action="store_true",
        default=False,
        help="비전 타워도 함께 학습할지 여부.",
    )
    ap.add_argument(
        "--target-modules",
        default="all-linear",
        help='기본값 "all-linear"는 Unsloth가 적절한 linear 레이어만 선택.',
    )
    ap.add_argument(
        "--use-kernels",
        action="store_true",
        default=True,
        help="Hub Kernels로 GatedDeltaNet 연산을 교체.",
    )
    ap.add_argument("--mlflow-experiment", default="qwen36-27b-task-loras")
    args = ap.parse_args()

    import mlflow

    mlflow.set_experiment(args.mlflow_experiment)

    model, processor = FastModel.from_pretrained(
        args.base_model,
        load_in_4bit=False,
        max_seq_length=args.max_seq_len,
        gpu_memory_utilization=0.8,
        attn_implementation="sdpa",
        full_finetuning=False,
        use_kernels=True,
        device_map={"": torch.cuda.current_device()},
    )
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    target_modules = (
        "all-linear"
        if args.target_modules == "all-linear"
        else [m.strip() for m in args.target_modules.split(",")]
    )

    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=args.finetune_vision,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        use_gradient_checkpointing=True,
        bias="none",
        random_state=3407,
        use_rslora=False,
    )

    for config in [model.config, getattr(model, "generation_config", None)]:
        if config is not None:
            config.pad_token_id = tokenizer.pad_token_id
            config.bos_token_id = tokenizer.bos_token_id
            config.eos_token_id = tokenizer.eos_token_id

    model.print_trainable_parameters()

    ds = load_dataset(
        "json",
        data_files={
            "train": f"{args.data_root}/{args.task}_train.jsonl",
            "validation": f"{args.data_root}/{args.task}_val.jsonl",
        },
    )
    ds["train"] = ds["train"].select(range(500))
    ds["validation"] = ds["validation"].select(range(25))

    def clean_tools(tools):
        """tools 데이터를 검증하고 유효한 JSON schema(dict) 목록으로 정제"""
        if not tools:
            return None
        if isinstance(tools, str):
            try:
                tools = json.loads(tools)
            except Exception:
                return None
        if isinstance(tools, dict):
            return [tools]
        if isinstance(tools, list):
            cleaned = []
            for t in tools:
                if not t:
                    continue
                if isinstance(t, str):
                    try:
                        t = json.loads(t)
                    except Exception:
                        continue
                if isinstance(t, dict) or callable(t):
                    cleaned.append(t)
            return cleaned if cleaned else None
        if callable(tools):
            return [tools]
        return None

    def format_example(examples):
        messages_field = examples["messages"]
        tools_field = examples.get("tools")

        is_batched = (
            isinstance(messages_field, list)
            and len(messages_field) > 0
            and isinstance(messages_field[0], list)
        )

        if is_batched:
            tools_list = (
                tools_field if tools_field is not None else [None] * len(messages_field)
            )
            formatted_texts = []
            for messages, tools in zip(messages_field, tools_list):
                c_tools = clean_tools(tools)
                kwargs = {"tokenize": False, "add_generation_prompt": False}
                if c_tools is not None:
                    kwargs["tools"] = c_tools
                formatted_texts.append(
                    tokenizer.apply_chat_template(messages, **kwargs)
                )
            return formatted_texts
        else:
            c_tools = clean_tools(tools_field)
            kwargs = {"tokenize": False, "add_generation_prompt": False}
            if c_tools is not None:
                kwargs["tools"] = c_tools
            return [tokenizer.apply_chat_template(messages_field, **kwargs)]

    FastModel.for_training(model)

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        max_length=args.max_seq_len,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=args.epochs,
        auto_find_batch_size=False,
        learning_rate=args.lr,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        optim="adamw_8bit",
        warmup_steps=5,
        lr_scheduler_type="linear",
        logging_steps=10,
        report_to=["mlflow"],
        loss_type="nll",
    )

    with mlflow.start_run(run_name=f"{args.task}-lora-r{args.lora_r}"):
        mlflow.log_params(
            {
                "task": args.task,
                "base_model": args.base_model,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "target_modules": str(target_modules),
                "finetune_vision": args.finetune_vision,
                "trainer": "unsloth",
            }
        )
        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=ds["train"],
            eval_dataset=ds["validation"],
            formatting_func=format_example,
            processing_class=processor,
        )
        trainer.train()
        trainer.save_model(args.output_dir)
        processor.save_pretrained(args.output_dir)

        mlflow.log_artifacts(args.output_dir, artifact_path=f"lora_adapter_{args.task}")

    print(f"Saved LoRA adapter -> {args.output_dir}")


if __name__ == "__main__":
    main()
