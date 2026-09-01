# FP8 KV Cache for Long-Context Serving

**Status:** Planning

## Question

Does FP8 KV cache let vLLM serve more long-context requests without causing too much quality loss?

We will test `Qwen/Qwen3-8B` on one NVIDIA H100 at 4K, 16K, and 32K context lengths.

## What we will compare

| ID | Weights | KV cache | Scales |
|---|---|---|---|
| C0 | BF16 | BF16 | N/A |
| C1 | BF16 | FP8 E4M3 | Default, uncalibrated |
| C2 | BF16 | FP8 E4M3 | Calibrated per tensor |

Model weights stay in BF16 for every run.

Per-head calibration and mixed BF16/FP8 layers are optional follow-up experiments. We will add them only if C1 or C2 shows a clear problem.

## What we will measure

### Serving

- Maximum concurrent sessions
- Sustainable requests per second
- SLO-compliant requests per second
- Time to first token (TTFT)
- Time per output token (TPOT)
- End-to-end latency
- Token throughput
- KV-cache utilization
- GPU-memory utilization
- Preemptions, failures, and recomputed tokens

### Quality

- Accuracy at 4K, 16K, and 32K
- Accuracy by task type
- BF16-correct examples that become incorrect with FP8
- Variation across repeated runs

Quality tests and load tests will be run separately.

## Quality datasets

We will use:

- Frozen RULER subsets for controlled retrieval, tracking, and aggregation tests
- A frozen LongBench-v2 subset for realistic long-context tasks

Calibration examples cannot appear in either evaluation set.

## Serving workloads

We will build the workloads in this order:

1. Unique 4K requests
2. Unique 16K requests
3. Unique 32K requests
4. Shared-prefix 16K requests
5. Multi-turn sessions that grow from 16K to 32K
6. A fixed mixed workload

Every prompt will be tokenized with the exact model tokenizer. The verified token count and prompt hash will be saved before testing begins.

Load tests will use fixed open-loop arrival traces. Requests will still arrive at their scheduled time when the server is overloaded.

Example trace row:

```json
{
  "arrival_ms": 0,
  "request_id": "request-0001",
  "session_id": null,
  "turn_id": null,
  "prompt_file": "prompts/cold_16k/request-0001.txt",
  "prompt_tokens": 16000,
  "max_tokens": 128,
  "traffic_class": "cold_unique"
}
```

## Development setup

### Development runs

- SageMaker Code Editor
- NVIDIA L4 (`ml.g6`)
- Small Qwen3 model
- 1K and 4K contexts

These runs will validate the code, request flow, timestamps, scoring, and result files. Their performance numbers will not be used as H100 results.

### Final runs

- One NVIDIA H100 (`ml.p5.4xlarge` or equivalent)
- `Qwen/Qwen3-8B`
- 4K, 16K, and 32K contexts
- Same frozen prompts and traces for C0, C1, and C2

Using one GPU avoids tensor-parallel communication during the main comparison.

## Controls

The following settings must stay fixed between comparable runs:

- Model and tokenizer revisions
- Weight dtype
- vLLM version
- CUDA runtime and NVIDIA driver
- Attention backend
- GPU type
- Maximum model length
- GPU-memory-utilization setting
- Prefix-caching setting
- Scheduler and batching settings
- Prompts and arrival traces
- Sampling settings
- Output-token limits
- Seeds, warm-up requests, and repetitions

The vLLM server will be restarted after changing the KV-cache condition.

Some vLLM attention backends perform more than FP8 storage when FP8 KV cache is enabled. For example, FlashAttention-3 can also quantize queries and run attention operations in FP8. We will record the backend and describe the tested execution path precisely.

## Memory estimate

We will calculate expected KV-cache memory before running the benchmark:

```text
head_dimension = hidden_size / num_attention_heads

KV_bytes_per_token =
    2
    × num_hidden_layers
    × num_key_value_heads
    × head_dimension
    × bytes_per_element
```

The factor `2` represents keys and values.

- BF16: 2 bytes per element
- FP8: 1 byte per element

This estimates tensor storage. Runtime capacity will also depend on weights, activations, workspaces, cache blocks, and fragmentation.

## Initial success rule

We will freeze the final thresholds before running the full FP8 comparison.

The current target is:

- At least 1.5× BF16 SLO goodput or sustainable 32K-session capacity
- At least 98% aggregate BF16 quality
- No primary task loses more than 2 absolute percentage points

A result can still be useful if FP8 misses these targets. We will report what happened and where it failed.

## Build order

1. Inspect the model configuration and calculate expected KV memory.
2. Start a BF16 vLLM server and send one streaming request.
3. Record prompt tokens, output tokens, dispatch time, first-token time, and completion time.
4. Generate and freeze one 4K workload.
5. Run a BF16 single-request context sweep.
6. Repeat the sweep with uncalibrated FP8.
7. Add open-loop load testing.
8. Run paired BF16 and FP8 quality tests.
9. Add calibrated FP8.
10. Repeat the final experiment on H100 and write the report.

## Result records

Each request will save:

- Request, session, and turn IDs
- Scheduled arrival and actual dispatch
- First-token and completion timestamps
- Prompt and output token counts
- Raw response
- Error or finish reason
- Traffic class
- Experiment condition

Each run will save:

- Git commit
- Model and tokenizer revisions
- Package versions
- GPU, driver, and CUDA information
- Attention backend
- Full server command
- Full benchmark configuration
- Workload hashes
- Seeds and warm-up settings

Raw results will never be overwritten.

## Repository layout

```text
fp8-kv-cache-study/
├── README.md
├── pyproject.toml
├── configs/
├── scripts/
├── src/fp8_kv_study/
├── tests/
├── notebooks/
├── workloads/
├── results/
│   ├── raw/
│   ├── processed/
│   └── figures/
└── reports/
```

Downloaded models, datasets, credentials, and large result files will not be committed.

## Output

The finished repository will contain:

- Reproducible benchmark code
- Frozen workloads
- Raw request logs
- BF16 and FP8 comparisons
- Calibration ablation
- Capacity, latency, and quality plots
- Failure analysis
- Reproduction instructions
- Final report with measured results and limitations

## Out of scope

- Weight quantization
- Model training or fine-tuning
- Custom CUDA or Triton kernels
- Multi-GPU performance
- Production endpoint deployment
- Claims about untested models or hardware

## References

- [vLLM Quantized KV Cache](https://docs.vllm.ai/en/stable/features/quantization/quantized_kvcache/)
- [vLLM FP8 KV-cache study](https://vllm.ai/blog/2026-04-22-fp8-kvcache)
- [LLM Compressor KV-cache example](https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/quantization_kv_cache/)
- [NVIDIA RULER](https://github.com/NVIDIA/RULER)
- [LongBench](https://github.com/THUDM/LongBench)
