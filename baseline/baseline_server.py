import argparse

from sglang.test.doc_patch import launch_server_cmd
from sglang.utils import wait_for_server, print_highlight, terminate_process

import asyncio
import openai
from pathlib import Path
import csv
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm
import logging

languages = {
    'fra': 'French',
    'ber': 'Berber',
    'cmn': 'Mandarin Chinese',
    'deu': 'German',
    'ita': 'Italian',
    'jpn': 'Japanese',
    'tur': 'Turkey'
}

python_path = "/home/export/doriancl/code/Fall-2025-ML-proj/.venv/bin/python"


def create_prompts(dataset: Path, n_samples: int | None = None, target_lang: str = "French"):
    """
    Read the CSV, take the first n_samples rows, and create prompts.
    Returns a list of dicts: {english, reference, prompt}.
    Assumes headers: English,Translation
    """
    samples = []

    with dataset.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if n_samples and len(samples) >= n_samples:
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


def launch_server(model: str) -> tuple:
    # launch the sglang server
    server_process, port = launch_server_cmd(
        f"""
    {python_path} -m sglang.launch_server --model-path {model} \
     --host 0.0.0.0 --log-level warning
    """
    )

    wait_for_server(f"http://localhost:{port}")
    print_highlight(f"SGLang server is up on port {port}")
    return server_process, port


async def run_batch(
        dataset_path: str, output_path: str,
        target_language: str,
        port,
        max_concurrent: int = 512, n_samples: int | None = None):
    """
    Run n_samples translations with bounded concurrency.
    max_concurrent controls how many HTTP requests are in flight at once.
    Higher -> more batching & throughput; too high -> possible OOM / timeouts.
    """
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    client = openai.AsyncClient(
        base_url=f"http://127.0.0.1:{port}/v1",
        api_key="None",
    )

    samples = create_prompts(dataset_path, n_samples=n_samples, target_lang=target_language)

    # Storage for responses aligned with samples by index
    responses = [None] * len(samples)

    # Semaphore to bound concurrency
    sem = asyncio.Semaphore(max_concurrent)

    # Progress bars
    submit_bar = tqdm(total=len(samples), desc="Submitting requests")
    recv_bar = tqdm(total=len(samples), desc="Receiving responses")

    async def worker(i: int, sample: dict):
        """
        One logical request: acquires the semaphore, submits the HTTP call,
        and records the response.
        """
        async with sem:
            submit_bar.update(1)
            try:
                resp = await client.chat.completions.create(
                    model="qwen/qwen2.5-0.5b-instruct",
                    messages=[{"role": "user", "content": sample["prompt"]}],
                    temperature=0,
                    max_tokens=128,
                )
            except Exception as e:
                print(f"Error in request {i}: {e}")
                resp = None

            responses[i] = resp
            recv_bar.update(1)

    # Create all tasks at once; semaphore enforces true concurrency cap
    tasks = [
        asyncio.create_task(worker(i, sample))
        for i, sample in enumerate(samples)
    ]

    try:
        await asyncio.gather(*tasks)
    finally:
        submit_bar.close()
        recv_bar.close()
        await client.close()

    # --- WRITE CSV ---
    with output_path.open("w", newline="", encoding="utf-8") as f_out:
        fieldnames = ["English", "AI_Translation", "Original_Translation"]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for sample, resp in zip(samples, responses):
            ai_text = resp.choices[0].message.content.strip() if resp else "[ERROR]"
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


async def amain(language: str, server_process, port, model: str):
    # --- config for this experiment ---
    # dataset_path = Path(f"/home/export/doriancl/code/Fall-2025-ML-proj/data/tatoeba/{language}.csv")
    # output_path = Path(f"/home/export/doriancl/code/Fall-2025-ML-proj/data/tatoeba/{language}_ai_translations.csv")
    #
    # # Change this to whatever language you want the model to translate into
    # target_language = languages[language]  # e.g. "French", "German", "Spanish", etc.

    config = {
        'dataset_path': Path(f"/home/export/doriancl/code/Fall-2025-ML-proj/data/tatoeba/{language}.csv"),
        'output_path': Path(f"/home/export/doriancl/code/Fall-2025-ML-proj/data/tatoeba/output/{model.lower()}/{language}_ai_translations.csv"),

        # Change this to whatever language you want the model to translate into
        'target_language': languages[language],  # e.g. "French", "German", "Spanish", etc.
        'max_concurrent': 1024,
        'port': port,
    }

    try:
        await run_batch(**config)
    finally:
        # Make sure we shut down the server process
        terminate_process(server_process)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', type=str)
    parser.add_argument('--model', type=str, default='qwen/qwen2.5-0.5b-instruct')

    args = parser.parse_args()

    server_process, port = launch_server(args.model)

    asyncio.run(amain(language=args.language,
                      server_process=server_process,
                      port=port,
                      model=args.model))
