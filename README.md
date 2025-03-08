# script-for-llm-distillation-generation


1) run `create.py`;
2) run `concat.py`
3) run `check.py`

! In the `llama-factory` implementation, the function `Trainer.prepare_input` casts floating-point tensors to the `bf16` data type (to align the dtype with LLM model), which may lead to overflow issues (e.g., `0.995` in `fp32` cast into `1.0` in `bf16`).
