from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


MODEL_ID = "Qwen/Qwen3-1.7B"
TARGET_TOKENS = 4096

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

dataset = load_dataset(
    "Salesforce/wikitext",
    "wikitext-103-raw-v1",
    split="train",
)

parts = []
token_ids = []

for row in dataset:
    text = row["text"].strip()

    if not text:
        continue

    parts.append(text)
    combined_text = "\n\n".join(parts)
    token_ids = tokenizer.encode(
        combined_text,
        add_special_tokens=False,
    )

    if len(token_ids) >= TARGET_TOKENS:
        break

prompt_ids = token_ids[:TARGET_TOKENS]

prompt_text = tokenizer.decode(
    prompt_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)

verified_count = len(
    tokenizer.encode(prompt_text, add_special_tokens=False)
)

repo_root = Path(__file__).resolve().parents[1]
output_path = repo_root / "workloads/prompts/cold_4k.txt"
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(prompt_text, encoding="utf-8")

print(f"Saved: {output_path}")
print(f"Verified tokens: {verified_count}")

if verified_count != TARGET_TOKENS:
    raise ValueError(
        f"Expected {TARGET_TOKENS} tokens, got {verified_count}"
    )