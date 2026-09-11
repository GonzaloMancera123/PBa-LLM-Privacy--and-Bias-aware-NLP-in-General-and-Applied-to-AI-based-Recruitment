# -*- coding: utf-8 -*-
"""
BATCH anonymization script using LOCAL Ollama models. Loops through
Qwen3 -> Gemma3 -> Phi-4-mini, each with both PER and LOC entity types
(6 combinations total), fully automatically.

No API key needed for any of these -- everything runs locally via
Ollama. Make sure the models are pulled first:
    ollama pull qwen3:8b
    ollama pull gemma3:4b
    ollama pull phi4-mini

Each (model, entity) combination is resumable on its own (thanks to
incremental saving) and wrapped in a try/except, so if one combination
fails partway through, the script logs it and moves on to the next
one instead of stopping entirely.
"""
import os
import traceback
import numpy as np
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Number of concurrent requests sent to Ollama. Must be <= the
# OLLAMA_NUM_PARALLEL value the server was started with (see setup
# instructions). A GTX 1070 Ti-class GPU might only handle 2; an
# RTX 4090 with an 8B model can comfortably handle 4-8.
MAX_WORKERS = 4


client = OpenAI(
    api_key="ollama",  # placeholder, Ollama does not check this
    base_url="http://localhost:11434/v1",  # GPU 0 instance
)

PROMPTS = {
    "PER": """Act as a name anonymizer. Replace proper names of people in the sentences with [MASK], without modifying titles, pronouns, or other parts of the sentence. Do not alter city names, articles, or prepositions, only personal names. Provide only the modified sentence without explanations or quotation marks.

Examples:
- "John met Sarah at the coffee shop" -> "[MASK] met [MASK] at the coffee shop"
- "Emma and Robert went on vacation" -> "[MASK] and [MASK] went on vacation"

Your turn: {text}
""",
    "LOC": """Act as a location anonymizer. Replace proper names of cities, countries, regions, or other geographic locations in the sentences with [MASK], without modifying personal names, titles, pronouns, or other parts of the sentence. Provide only the modified sentence without explanations or quotation marks.

Examples:
- "She was born in Boston and moved to Madrid" -> "She was born in [MASK] and moved to [MASK]"
- "The conference took place in Berlin last year" -> "The conference took place in [MASK] last year"

Your turn: {text}
""",
}

# ============================================================
# All 6 combinations to run, in order.
# ============================================================
CONFIGURATIONS = [


        {"model": "gemma3:4b", "entity": "PER", "tag": "GEMMA3"},
    {"model": "gemma3:4b", "entity": "LOC", "tag": "GEMMA3"},
]

data_path = r"C:\Users\Puesto-2\Documents\Gonzalo\NLP\NLPCV\MIA\data"
database_file = "FairCVdb.npy"
ANON_ROOT = r"C:\Users\Puesto-2\Documents\Gonzalo\NLP\NLPCV\Anonimization"


def anonymize_one(original_text, prompt_template, model_name):
    """Sends a single anonymization request. Used by the thread pool."""
    input_text = prompt_template.format(text=original_text)
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": input_text}],
            max_tokens=150,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return None  # signals failure; caller falls back to original_text


def anonymize_split(bios_full, output_path, prompt_template, model_name, desc=""):
    """Anonymizes a full Bios array using MAX_WORKERS concurrent requests,
    processed in order-preserving chunks. Resumable via incremental
    saving after each chunk (not after every single row, to keep
    throughput high -- at most one chunk of progress is lost if
    interrupted mid-chunk)."""
    if os.path.exists(output_path):
        anonymized = list(np.load(output_path, allow_pickle=True))
        start_index = len(anonymized)
        if start_index >= len(bios_full):
            print(f"  Already complete ({start_index}/{len(bios_full)}). Skipping.")
            return anonymized
        print(f"  Resuming from {start_index}/{len(bios_full)} rows already processed.")
    else:
        anonymized = []
        start_index = 0
        print("  No existing file found. Starting from scratch.")

    pbar = tqdm(total=len(bios_full), initial=start_index, desc=desc, unit="bio")

    i = start_index
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        while i < len(bios_full):
            chunk_end = min(i + MAX_WORKERS, len(bios_full))
            chunk_indices = list(range(i, chunk_end))
            chunk_texts = [bios_full[j][0] for j in chunk_indices]

            # Fire all requests in this chunk concurrently, but collect
            # results back in the original order.
            futures = {executor.submit(anonymize_one, text, prompt_template, model_name): pos
                       for pos, text in enumerate(chunk_texts)}
            results = [None] * len(chunk_texts)
            for future in as_completed(futures):
                pos = futures[future]
                results[pos] = future.result()

            for pos, original_text in enumerate(chunk_texts):
                anonymized_text = results[pos] if results[pos] is not None else original_text
                anonymized.append([original_text, anonymized_text])

            np.save(output_path, anonymized)
            pbar.update(len(chunk_indices))
            i = chunk_end

    pbar.close()
    return anonymized


# ============================================================
# Run all 6 combinations
# ============================================================
if __name__ == "__main__":

    fairCV = np.load(os.path.join(data_path, database_file), allow_pickle=True).item()
    bios_train_full = fairCV['Bios Train']  # 19200 rows
    bios_test_full = fairCV['Bios Test']    # 4800 rows

    completed = []
    failed = []

    for cfg in CONFIGURATIONS:
        model_name = cfg["model"]
        entity = cfg["entity"]
        tag = cfg["tag"]

        print("\n" + "=" * 70)
        print(f"CONFIGURATION: {tag} ({model_name}) - Entity: {entity}")
        print("=" * 70)

        try:
            save_root = os.path.join(ANON_ROOT, tag)
            train_save_path = os.path.join(save_root, "Train")
            test_save_path = os.path.join(save_root, "Test")
            for d in [train_save_path, test_save_path]:
                if not os.path.isdir(d):
                    os.makedirs(d)

            train_output_path = os.path.join(train_save_path, f"anonymized_fairCV{entity}.npy")
            test_output_path = os.path.join(test_save_path, f"anonymized_fairCV{entity}.npy")

            prompt_template = PROMPTS[entity]

            print(f"\n--- Anonymizing TRAIN (all {len(bios_train_full)} rows) ---")
            print(f"Output: {train_output_path}")
            anonymize_split(bios_train_full, train_output_path, prompt_template, model_name,
                             desc=f"{tag}-{entity} TRAIN")

            print(f"\n--- Anonymizing TEST (all {len(bios_test_full)} rows) ---")
            print(f"Output: {test_output_path}")
            anonymize_split(bios_test_full, test_output_path, prompt_template, model_name,
                             desc=f"{tag}-{entity} TEST")

            print(f"\nCompleted: {tag} - {entity}")
            completed.append(f"{tag}_{entity}")

        except Exception as e:
            print(f"  !! FAILED: {tag} - {entity}: {e}")
            traceback.print_exc()
            failed.append(f"{tag}_{entity}")

    # ============================================================
    # Final summary
    # ============================================================
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\n✅ Completed ({len(completed)}): {completed}")
    if failed:
        print(f"\n❌ Failed ({len(failed)}): {failed}")