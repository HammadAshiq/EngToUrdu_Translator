"""
Final data rebalancing step.

Combines everything gathered so far into one balanced training corpus:
  - OPUS: OpenSubtitles, Tanzil (CAPPED -- see below), QED, GNOME, KDE4, Ubuntu
  - BPCC: bpcc-seed-latest + wiki subsets (human-curated, general + encyclopedic domain)

The original run was ~97% Tanzil (Quranic, formal/religious register), which is
why casual conversational input translated poorly. This script caps Tanzil to
a minority share so the model sees a realistic mix of everyday, encyclopedic,
and formal text instead of being dominated by one narrow domain.

Run from the project root (same folder as download_data.py, transformer.py, etc).
Requires data/raw/{OpenSubtitles,Tanzil,QED,GNOME,KDE4,Ubuntu}/corpus.{en,ur}
(from download_data.py) and data/raw/BPCC/{bpcc-seed-latest,wiki}/urd_Arab.tsv
(from the hf download commands) to already exist.
"""

import csv
import os
import random

MAX_TANZIL = 100_000  # cap -- was ~723k (97% of the old corpus), now a minority
MAX_SENT_LEN = 80
MAX_LEN_RATIO = 3.0
SEED = 42

random.seed(SEED)


def sanitize(text: str) -> str:
    """Collapse any embedded newline/tab/carriage-return characters to a
    single space. Without this, a sentence containing a literal '\\n' would
    silently turn into two lines when the corpus is written out with
    '\\n'.join(...), misaligning the English and Urdu files relative to
    each other even though they started out perfectly paired."""
    return " ".join(text.split())


def load_opus_pair(name: str):
    en_path = f"data/raw/{name}/corpus.en"
    ur_path = f"data/raw/{name}/corpus.ur"
    if not (os.path.exists(en_path) and os.path.exists(ur_path)):
        print(f"  [{name}] not found, skipping")
        return [], []
    with open(en_path, encoding="utf-8") as f:
        en = [sanitize(l) for l in f if l.strip()]
    with open(ur_path, encoding="utf-8") as f:
        ur = [sanitize(l) for l in f if l.strip()]
    n = min(len(en), len(ur))
    return en[:n], ur[:n]


def load_bpcc_tsv(path: str):
    """Parses by column name ('src' = English, 'tgt' = Urdu) since the two
    BPCC subsets use different column orders in their TSVs."""
    if not os.path.exists(path):
        print(f"  [{path}] not found, skipping")
        return [], []
    en, ur = [], []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if "src" not in reader.fieldnames or "tgt" not in reader.fieldnames:
            print(f"  [{path}] unexpected columns {reader.fieldnames}, skipping")
            return [], []
        for row in reader:
            s, t = sanitize(row["src"]), sanitize(row["tgt"])
            if s and t:
                en.append(s)
                ur.append(t)
    return en, ur


def main():
    all_en, all_ur, source_counts = [], [], {}

    print("Loading OPUS corpora...")
    for name in ["OpenSubtitles", "QED", "GNOME", "KDE4", "Ubuntu"]:
        en, ur = load_opus_pair(name)
        print(f"  [{name}] {len(en)} pairs")
        source_counts[name] = len(en)
        all_en.extend(en)
        all_ur.extend(ur)

    print(f"\nLoading Tanzil (capping at {MAX_TANZIL})...")
    tanzil_en, tanzil_ur = load_opus_pair("Tanzil")
    print(f"  [Tanzil] {len(tanzil_en)} raw pairs")
    if len(tanzil_en) > MAX_TANZIL:
        idx = list(range(len(tanzil_en)))
        random.shuffle(idx)
        idx = idx[:MAX_TANZIL]
        tanzil_en = [tanzil_en[i] for i in idx]
        tanzil_ur = [tanzil_ur[i] for i in idx]
    print(f"  [Tanzil] {len(tanzil_en)} pairs after capping")
    source_counts["Tanzil (capped)"] = len(tanzil_en)
    all_en.extend(tanzil_en)
    all_ur.extend(tanzil_ur)

    print("\nLoading BPCC subsets...")
    for name, path in [
        ("BPCC-seed", "data/raw/BPCC/bpcc-seed-latest/urd_Arab.tsv"),
        ("BPCC-wiki", "data/raw/BPCC/wiki/urd_Arab.tsv"),
    ]:
        en, ur = load_bpcc_tsv(path)
        print(f"  [{name}] {len(en)} pairs")
        source_counts[name] = len(en)
        all_en.extend(en)
        all_ur.extend(ur)

    print(f"\nTotal before cleaning: {len(all_en)} pairs")
    print("\nSource breakdown:")
    total = sum(source_counts.values())
    for name, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total if total else 0
        print(f"  {name:20s} {count:>8,}  ({pct:5.1f}%)")

    # clean: dedup, length ratio filter, max length
    seen = set()
    clean_en, clean_ur = [], []
    for en, ur in zip(all_en, all_ur):
        if not en or not ur:
            continue
        len_en, len_ur = len(en.split()), len(ur.split())
        if len_en == 0 or len_ur == 0:
            continue
        if max(len_en, len_ur) / min(len_en, len_ur) > MAX_LEN_RATIO:
            continue
        if len_en > MAX_SENT_LEN or len_ur > MAX_SENT_LEN:
            continue
        key = (en, ur)
        if key in seen:
            continue
        seen.add(key)
        clean_en.append(en)
        clean_ur.append(ur)

    print(f"\nTotal after cleaning: {len(clean_en)} pairs")

    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/train.en", "w", encoding="utf-8") as f:
        f.write("\n".join(clean_en))
    with open("data/processed/train.ur", "w", encoding="utf-8") as f:
        f.write("\n".join(clean_ur))

    print("\nWrote data/processed/train.en and data/processed/train.ur")
    print("\nNext steps:")
    print("  1. Re-run train_tokenizer.py -- vocab should reflect the new, more balanced mix")
    print("  2. Re-run dataset.py as a smoke test")
    print("  3. Start a fresh training run in train_transformer.ipynb (old checkpoints/tokenizers "
          "won't match this new data, so this is a clean restart, not a resume)")


if __name__ == "__main__":
    main()