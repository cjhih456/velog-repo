# llm-finetune-sample

Qwen3.6-27B를 Hermes 에이전트 추론 트레이스로 LoRA 파인튜닝하고, NVFP4 양자화 모델 위에 어댑터를 올려 vLLM으로 서빙하는 실습 샘플입니다.

학습은 Base Model에서, 서빙은 양자화 모델 + LoRA adapter로 분리합니다.

| 구분 | 리소스 |
| --- | --- |
| Base Model | [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) |
| Serving Model | [nvidia/Qwen3.6-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.6-27B-NVFP4) |
| Dataset | [lambda/hermes-agent-reasoning-traces](https://huggingface.co/datasets/lambda/hermes-agent-reasoning-traces) |

## 파이프라인

```
hermes-agent-reasoning-traces  →  01c 변환  →  01d 검증  →  02 LoRA 학습  →  99 vLLM 서빙
     (ShareGPT)                    JSONL          template          Unsloth SFT       NVFP4 + adapter
```

| 파일 | 역할 |
| --- | --- |
| `01c_convert_herems_traces.py` | ShareGPT → OpenAI messages 포맷 변환, chat template 검증, train/val 분리 |
| `01d_spotcheck_template.py` | 변환된 JSONL이 Qwen chat template로 정상 렌더링되는지 샘플 확인 |
| `02_train_lora_2.py` | Unsloth + SFTTrainer로 16bit LoRA 학습 |
| `99_vllm_start.sh` | NVFP4 양자화 모델에 LoRA adapter를 붙여 vLLM 서빙 |

실행 순서와 환경 설치는 상위 문서 `llm-finetuning.md`를 참고하세요.
