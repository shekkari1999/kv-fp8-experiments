import json
import time
from pathlib import Path

import requests


repo_root = Path(__file__).resolve().parents[1]
prompt_path = repo_root / "workloads/prompts/cold_4k.txt"

URL = "http://127.0.0.1:8000/v1/completions"


def run_request():
    prompt = prompt_path.read_text(encoding="utf-8")

    payload = {
        "model": "Qwen/Qwen3-1.7B",
        "prompt": prompt,
        "max_tokens": 128,
        "temperature": 0,
        "stream": True,
        "stream_options": {
            "include_usage": True,
        },
    }

    start_time = time.perf_counter()
    first_token_time = None
    output_tokens = None
    chunks = []

    with requests.post(
        URL,
        json=payload,
        stream=True,
        timeout=300,
    ) as response:
        response.raise_for_status()

        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue

            data = line.removeprefix("data: ")

            if data == "[DONE]":
                break

            event = json.loads(data)

            if event.get("usage"):
                output_tokens = event["usage"]["completion_tokens"]

            if not event.get("choices"):
                continue

            text = event["choices"][0]["text"]

            if text:
                if first_token_time is None:
                    first_token_time = time.perf_counter()

                chunks.append(text)

    end_time = time.perf_counter()

    if first_token_time is None or output_tokens is None:
        raise RuntimeError("The server returned incomplete timing information.")

    return {
        "ttft_s": first_token_time - start_time,
        "tpot_s": (end_time - first_token_time) / (output_tokens - 1),
        "e2e_s": end_time - start_time,
        "output_tokens": output_tokens,
        "text": "".join(chunks),
    }


if __name__ == "__main__":
    result = run_request()

    print(f"TTFT: {result['ttft_s']:.4f} seconds")
    print(f"TPOT: {result['tpot_s']:.4f} seconds")
    print(f"E2E: {result['e2e_s']:.4f} seconds")
    print(f"Output tokens: {result['output_tokens']}")