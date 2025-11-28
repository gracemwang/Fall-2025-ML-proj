from sglang.test.doc_patch import launch_server_cmd
from sglang.utils import wait_for_server, print_highlight, terminate_process

import asyncio
import openai
from pathlib import Path
import csv

python_path = "/home/export/doriancl/code/Fall-2025-ML-proj/.venv/bin/python"

# --- config for this experiment ---
dataset_path = Path("/home/export/doriancl/code/Fall-2025-ML-proj/data/tatoeba/fra.csv")
output_path = Path("/home/export/doriancl/code/Fall-2025-ML-proj/data/tatoeba/fra_ai_translations.csv")

# Change this to whatever language you want the model to translate into
target_language = "French"  # e.g. "French", "German", "Spanish", etc.


def create_prompts(dataset: Path, n_samples: int = 1000, target_lang: str = "French"):
    """
    Read the CSV, take the first n_samples rows, and create prompts.
    Returns a list of dicts: {english, reference, prompt}.
    Assumes headers: English,Translation
    """
    samples = []

    with dataset.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if len(samples) >= n_samples:
                break

            english = (row.get("English") or "").strip()
            reference = (row.get("Translation") or "").strip()

            if not english:
                continue

            prompt = (
                f"Translate the following English sentence into {target_lang}. "
                "Respond with only the translation, no explanations:\n\n"
                f"{english}"
            )

            samples.append(
                {
                    "english": english,
                    "reference": reference,
                    "prompt": prompt,
                }
            )

    # If you *require* exactly 1000, you can assert here:
    if len(samples) != n_samples:
        print_highlight(f"Warning: only found {len(samples)} samples (requested {n_samples}).")

    return samples


# --- launch the sglang server (same as before) ---
server_process, port = launch_server_cmd(
    f"""
{python_path} -m sglang.launch_server --model-path qwen/qwen2.5-0.5b-instruct \
 --host 0.0.0.0 --log-level warning
"""
)

wait_for_server(f"http://localhost:{port}")
print_highlight(f"SGLang server is up on port {port}")


async def run_batch():
    # Async client talking to the sglang OpenAI-compatible endpoint
    client = openai.AsyncClient(
        base_url=f"http://127.0.0.1:{port}/v1",
        api_key="None",
    )

    # Build prompts from the first 1000 rows of the dataset
    samples = create_prompts(dataset_path, n_samples=200_000, target_lang=target_language)

    # Create all the request coroutines
    coros = [
        client.chat.completions.create(
            model="qwen/qwen2.5-0.5b-instruct",
            messages=[{"role": "user", "content": sample["prompt"]}],
            temperature=0,
            max_tokens=128,
        )
        for sample in samples
    ]

    try:
        # Run them concurrently
        responses = await asyncio.gather(*coros)

        # Write results to a new CSV: 3 columns
        with output_path.open("w", newline="", encoding="utf-8") as f_out:
            fieldnames = ["English", "AI_Translation", "Original_Translation"]
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()

            for sample, resp in zip(samples, responses):
                ai_text = resp.choices[0].message.content.strip()

                writer.writerow(
                    {
                        "English": sample["english"],
                        "AI_Translation": ai_text,
                        "Original_Translation": sample["reference"],
                    }
                )

        print_highlight(
            f"Done! Wrote {len(samples)} rows to {output_path} "
            f"(English, AI_Translation, Original_Translation)."
        )

    finally:
        # Close the HTTP client cleanly
        await client.close()


async def amain():
    try:
        await run_batch()
    finally:
        # Make sure we shut down the server process
        terminate_process(server_process)


if __name__ == "__main__":
    asyncio.run(amain())