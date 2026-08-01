"""
Step 1 of the data pipeline: download + clean EN-Urdu parallel data.

Two sources, combined:

1. OPUS -- downloaded here via the `opustools` package, which queries
   OPUS's live API for the correct, current download URL for each corpus
   (hardcoding URLs directly is fragile -- OPUS reshuffles file paths
   across corpus versions, so we let their own tool resolve it).

2. BPCC (manual download -- see instructions printed at the end of this
   script) -- AI4Bharat's Bharat Parallel Corpus Collection. This is the
   largest and highest-quality EN-Urdu resource currently available, but
   it's distributed from AI4Bharat's own site rather than a scriptable
   API, so grab it by hand once and drop it in data/raw/BPCC/.

Setup:
    pip install opustools tqdm
    python3 download_data.py

If a particular OPUS corpus name has changed or been retired, this script
will print a warning and skip it rather than fail outright -- check
https://opus.nlpl.eu/ for the current corpus list if that happens.
"""

import os
import subprocess

RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)

# OPUS collection names for EN-Urdu. Names must match opustools' collection
# list (see: opus_get --list or https://opus.nlpl.eu/opusapi/?corpora=True).
OPUS_COLLECTIONS = [
    # "OpenSubtitles",  # very large -- download the compiled en-ur bitext
                         # directly from https://opus.nlpl.eu/OpenSubtitles/en&ur/v2018/OpenSubtitles
                         # instead, then drop it in data/raw/OpenSubtitles/corpus.en(.ur).
                         # opus_read's auto-fetch can fall back to downloading the ENTIRE
                         # raw multi-language release if no compiled bitext exists for this pair.
    "Tanzil", "QED", "GNOME", "KDE4", "Ubuntu",
]


def download_opus():
    print("Downloading OPUS EN-Urdu corpora via opustools...\n")
    for name in OPUS_COLLECTIONS:
        out_dir = os.path.join(RAW_DIR, name)
        os.makedirs(out_dir, exist_ok=True)
        en_out = os.path.join(out_dir, "corpus.en")
        ur_out = os.path.join(out_dir, "corpus.ur")

        if os.path.exists(en_out) and os.path.exists(ur_out):
            print(f"[{name}] already downloaded, skipping")
            continue

        print(f"[{name}] fetching en-ur pair... (this can take several minutes for "
              f"large corpora like OpenSubtitles -- output streams live below)")
        cmd = [
            "opus_read",
            "--directory", name,
            "--source", "en",
            "--target", "ur",
            "--preprocess", "moses",
            "--write_mode", "moses",
            "--write", en_out, ur_out,
        ]
        result = subprocess.run(cmd)  # no capture -- prints straight to your terminal live

        if result.returncode != 0:
            print(f"  [SKIP] {name} failed -- see output above for the reason "
                  f"(missing en-ur pair, network issue, or changed collection name).")
            continue

        if os.path.exists(en_out):
            with open(en_out, encoding="utf-8") as f:
                n_lines = sum(1 for _ in f)
            print(f"  got {n_lines} lines")


def clean_and_merge():
    """Combine all downloaded corpora into two aligned files: train.en / train.ur
    Filters: drop empty lines, drop pairs with extreme length ratio (likely
    misaligned), drop exact duplicate pairs, drop overly long sentences.
    """
    en_lines, ur_lines = [], []

    for name in OPUS_COLLECTIONS + ["BPCC"]:
        out_dir = os.path.join(RAW_DIR, name)
        en_file = os.path.join(out_dir, "corpus.en")
        ur_file = os.path.join(out_dir, "corpus.ur")
        # BPCC manual drop-in uses BPCC.en / BPCC.ur -- check both naming options
        if name == "BPCC" and not os.path.exists(en_file):
            en_file = os.path.join(out_dir, "BPCC.en")
            ur_file = os.path.join(out_dir, "BPCC.ur")

        if not (os.path.exists(en_file) and os.path.exists(ur_file)):
            continue

        with open(en_file, encoding="utf-8") as f_en, open(ur_file, encoding="utf-8") as f_ur:
            en_content = f_en.readlines()
            ur_content = f_ur.readlines()

        if len(en_content) != len(ur_content):
            print(f"  [WARN] {name}: line count mismatch "
                  f"({len(en_content)} en vs {len(ur_content)} ur) -- skipping this corpus")
            continue

        print(f"[{name}] {len(en_content)} raw pairs")
        en_lines.extend(en_content)
        ur_lines.extend(ur_content)

    if not en_lines:
        print("\nNo data found yet. Run download_opus() first, and/or add BPCC manually "
              "(see instructions below).")
        return

    seen = set()
    clean_en, clean_ur = [], []
    for en, ur in zip(en_lines, ur_lines):
        en, ur = en.strip(), ur.strip()

        if not en or not ur:
            continue

        len_en, len_ur = len(en.split()), len(ur.split())
        if len_en == 0 or len_ur == 0:
            continue

        # crude length-ratio filter: catches badly misaligned sentence pairs
        ratio = max(len_en, len_ur) / min(len_en, len_ur)
        if ratio > 3.0:
            continue

        # drop very long sentences (helps training memory/speed on 6GB VRAM)
        if len_en > 80 or len_ur > 80:
            continue

        key = (en, ur)
        if key in seen:
            continue
        seen.add(key)

        clean_en.append(en)
        clean_ur.append(ur)

    print(f"\nTotal after cleaning: {len(clean_en)} pairs (from {len(en_lines)} raw pairs)")

    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/train.en", "w", encoding="utf-8") as f:
        f.write("\n".join(clean_en))
    with open("data/processed/train.ur", "w", encoding="utf-8") as f:
        f.write("\n".join(clean_ur))

    print("Wrote data/processed/train.en and data/processed/train.ur")


if __name__ == "__main__":
    download_opus()
    clean_and_merge()

    print(
        "\n"
        + "=" * 70 + "\n"
        "To add BPCC (recommended -- larger + higher quality than OPUS alone):\n"
        "1. Go to https://ai4bharat.iitm.ac.in/bpcc (search 'AI4Bharat BPCC download'\n"
        "   if that link has moved) and download the English-Urdu bitext files\n"
        "   (BPCC-Mined and/or BPCC-Human for en-ur)\n"
        "2. Place them as data/raw/BPCC/BPCC.en and data/raw/BPCC/BPCC.ur\n"
        "   (plain text, one sentence per line, aligned line-by-line)\n"
        "3. Re-run this script -- clean_and_merge() already looks for BPCC.en/.ur\n"
        + "=" * 70
    )