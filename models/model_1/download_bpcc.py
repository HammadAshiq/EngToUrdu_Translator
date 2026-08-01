"""
Download BPCC (AI4Bharat's Bharat Parallel Corpus Collection) for the
English-Urdu pair, plus the IN22-Conv conversational benchmark for a
proper held-out conversational-domain eval set.

IMPORTANT: I could not verify the exact Hugging Face config/column names
for these datasets from my sandbox (huggingface.co isn't reachable from
there). This script is written from AI4Bharat's documented structure, but
run the --inspect step first -- if anything doesn't match, paste me the
printed output and I'll fix the field names in one pass rather than you
guessing at it.

Setup:
    pip install datasets tqdm
    python3 download_bpcc.py --inspect     # just look at what's there first
    python3 download_bpcc.py               # actually download once confirmed
"""

import argparse
import os
import random

RAW_DIR = "data/raw/BPCC"
os.makedirs(RAW_DIR, exist_ok=True)

# cap on how much of BPCC-Mined to pull -- it can run into the millions for
# a language pair, and we don't need all of it. This many pairs, combined
# with everything else, keeps a full training run at a reasonable size
# even though you're fine with a long training time.
MINED_SAMPLE_CAP = 1_500_000


def inspect():
    from datasets import get_dataset_config_names, load_dataset

    print("Looking up available BPCC configs...")
    try:
        configs = get_dataset_config_names("ai4bharat/BPCC")
        urdu_configs = [c for c in configs if "urd" in c.lower()]
        print(f"Found {len(configs)} total configs:")
        for c in configs:
            print(f"  {c}")
        print(f"\n{len(urdu_configs)} of these mention Urdu directly.")
    except Exception as e:
        print(f"Could not list configs: {e}")
        print("Falling back to trying 'eng_Latn-urd_Arab' directly.")
        urdu_configs = ["eng_Latn-urd_Arab"]

    if not urdu_configs:
        if configs:
            print(f"\nNo config name mentions Urdu directly -- this likely means "
                  f"BPCC is organized differently than I assumed (e.g. one config "
                  f"per subset like 'daily'/'wiki', with language pairs as rows "
                  f"inside, rather than one config per language pair).")
            print(f"Inspecting the first config ('{configs[0]}') to see its actual "
                  f"structure...")
            urdu_configs = [configs[0]]
        else:
            print("No configs found and config listing failed -- check "
                  "https://huggingface.co/datasets/ai4bharat/BPCC in a browser "
                  "for the exact config name and tell me what you see.")
            return

    config_name = urdu_configs[0]
    print("\n--- Checking specific subsets for eng_Latn / urd_Arab alignment ---")
    for config_name in ["daily", "bpcc-seed-latest", "samanantar-filtered", "nllb-filtered"]:
        print(f"\n[{config_name}]")
        try:
            en_ds = load_dataset("ai4bharat/BPCC", config_name, split="eng_Latn")
            ur_ds = load_dataset("ai4bharat/BPCC", config_name, split="urd_Arab")
            print(f"  eng_Latn rows: {len(en_ds)}, urd_Arab rows: {len(ur_ds)}")
            print(f"  eng_Latn columns: {en_ds.column_names}")
            if len(en_ds) == len(ur_ds):
                print(f"  MATCH -- likely row-aligned. Sample pairs:")
                for i in [0, 1, min(2, len(en_ds) - 1)]:
                    print(f"    EN: {en_ds[i]}")
                    print(f"    UR: {ur_ds[i]}")
            else:
                print(f"  MISMATCH -- lengths differ, NOT safely row-aligned. "
                      f"Would need an id field to align properly, or skip this subset.")
        except Exception as e:
            print(f"  Could not load: {e}")
    return


def download():
    from datasets import load_dataset
    from tqdm import tqdm

    config_name = "eng_Latn-urd_Arab"
    print(f"Streaming BPCC config: {config_name}")
    print("(If this errors or the field names below look wrong, run with "
          "--inspect first and send me the output.)")

    ds = load_dataset("ai4bharat/BPCC", config_name, split="train", streaming=True)

    en_lines, ur_lines = [], []
    rng = random.Random(42)

    # reservoir-style cap: keep every example up to MINED_SAMPLE_CAP, then
    # randomly replace to approximate a uniform sample if the stream is
    # larger than the cap (avoids just keeping "the first N", which could
    # be biased toward whatever order the source data happens to be in)
    n_seen = 0
    for example in tqdm(ds, desc="BPCC en-urd"):
        # NOTE: field names below are my best guess from AI4Bharat's
        # documented schema (eng_Latn / urd_Arab, matching the config name
        # convention). If --inspect showed different column names, change
        # these two lines to match.
        en_text = example.get("eng_Latn") or example.get("sentence_eng_Latn")
        ur_text = example.get("urd_Arab") or example.get("sentence_urd_Arab")

        if not en_text or not ur_text:
            continue

        n_seen += 1
        if len(en_lines) < MINED_SAMPLE_CAP:
            en_lines.append(en_text)
            ur_lines.append(ur_text)
        else:
            j = rng.randint(0, n_seen - 1)
            if j < MINED_SAMPLE_CAP:
                en_lines[j] = en_text
                ur_lines[j] = ur_text

    print(f"\nCollected {len(en_lines)} pairs (from {n_seen} seen in the stream)")

    with open(os.path.join(RAW_DIR, "BPCC.en"), "w", encoding="utf-8") as f:
        f.write("\n".join(en_lines))
    with open(os.path.join(RAW_DIR, "BPCC.ur"), "w", encoding="utf-8") as f:
        f.write("\n".join(ur_lines))

    print(f"Wrote {RAW_DIR}/BPCC.en and {RAW_DIR}/BPCC.ur")

    # IN22-Conv -- held out separately as a conversational-domain eval set,
    # NOT merged into training data
    print("\nDownloading IN22-Conv (conversational eval set, held out)...")
    try:
        ds_conv = load_dataset("ai4bharat/IN22-Conv", split="test")
        en_field = "sentence_eng_Latn" if "sentence_eng_Latn" in ds_conv.column_names else "eng_Latn"
        ur_field = "sentence_urd_Arab" if "sentence_urd_Arab" in ds_conv.column_names else "urd_Arab"

        os.makedirs("data/eval", exist_ok=True)
        with open("data/eval/in22_conv.en", "w", encoding="utf-8") as f:
            f.write("\n".join(ds_conv[en_field]))
        with open("data/eval/in22_conv.ur", "w", encoding="utf-8") as f:
            f.write("\n".join(ds_conv[ur_field]))
        print(f"Wrote data/eval/in22_conv.en and .ur ({len(ds_conv)} sentences)")
    except Exception as e:
        print(f"Could not download IN22-Conv: {e}")
        print("Not critical -- training data is unaffected, we just won't "
              "have this specific eval set. Can retry separately later.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect", action="store_true",
                         help="Just inspect available configs/schema, don't download")
    args = parser.parse_args()

    if args.inspect:
        inspect()
    else:
        download()