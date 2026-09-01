import json
import statistics
from pathlib import Path

from request_once import run_request


WARMUP_REQUESTS = 3
MEASURED_REQUESTS = 20

repo_root = Path(__file__).resolve().parents[1]
output_path = repo_root / "results/raw/bf16_single_4k.jsonl"
output_path.parent.mkdir(parents=True, exist_ok=True)


print("Running warm-ups...")

for i in range(WARMUP_REQUESTS):
    run_request()
    print(f"Warm-up {i + 1}/{WARMUP_REQUESTS}")


results = []

print("\nRunning measured requests...")

with output_path.open("w", encoding="utf-8") as file:
    for i in range(MEASURED_REQUESTS):
        result = run_request()

        record = {
            "run": i + 1,
            "ttft_s": result["ttft_s"],
            "tpot_s": result["tpot_s"],
            "e2e_s": result["e2e_s"],
            "output_tokens": result["output_tokens"],
        }

        results.append(record)
        file.write(json.dumps(record) + "\n")

        print(
            f"Run {i + 1:02d}: "
            f"TTFT={record['ttft_s']:.4f}s  "
            f"TPOT={record['tpot_s']:.4f}s  "
            f"E2E={record['e2e_s']:.4f}s"
        )


print("\nBF16 single-request baseline")
print(f"Median TTFT: {statistics.median(r['ttft_s'] for r in results):.4f}s")
print(f"Median TPOT: {statistics.median(r['tpot_s'] for r in results):.4f}s")
print(f"Median E2E:  {statistics.median(r['e2e_s'] for r in results):.4f}s")
print(f"Saved: {output_path}")