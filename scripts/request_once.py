import time
import json
import requests

from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
prompt_path = repo_root / "workloads/prompts/cold_4k.txt"
prompt = prompt_path.read_text(encoding="utf-8")

url = "http://127.0.0.1:8000/v1/completions"

payload = {
     "model": "Qwen/Qwen3-1.7B",
    "prompt": "The capital of France is",
    "max_tokens": 16,
    "temperature": 0,
    "stream": True,
}

start_time = time.perf_counter()
first_token_time = None
chunks = []
with requests.post(url, json=payload, stream=True, timeout=300) as response:
    response.raise_for_status()

    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue

        data = line.removeprefix("data: ")

        if data == "[DONE]":
            break

        event = json.loads(data)
        text = event["choices"][0]["text"]

        if text:
            if first_token_time is None:
                first_token_time = time.perf_counter()

            chunks.append(text)

end_time = time.perf_counter()

ttft = first_token_time - start_time
e2e = end_time - start_time
generated_text = "".join(chunks)

print(f"TTFT: {ttft:.4f} seconds")
print(f"E2E: {e2e:.4f} seconds")
print(f"Text: {generated_text}")