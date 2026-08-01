#!/bin/bash
# Reorganizes the flat project directory into a versioned models/ structure
# for GitHub. Run this from INSIDE your project folder (after renaming it
# and reactivating your venv -- see the instructions that came with this).
#
# Safe to re-run -- uses mkdir -p and mv (won't error if already moved),
# but back up first if you're unsure: cp -r . ../backup_before_reorg

set -e  # stop on first error rather than continuing with a half-done reorg

echo "=== Reorganizing into models/model_2_rebalanced/ ==="
mkdir -p models/model_1_baseline
mkdir -p models/model_2_rebalanced

# --- move the current (working, rebalanced) code into model_2 ---
for f in transformer.py dataset.py download_data.py download_bpcc.py \
         rebalance_data.py train_tokenizer.py train_transformer.ipynb \
         test_transformer.py; do
    if [ -f "$f" ]; then
        mv "$f" models/model_2_rebalanced/
        echo "  moved $f -> models/model_2_rebalanced/"
    else
        echo "  [skip] $f not found in current directory"
    fi
done

# tokenizers and checkpoints belong to model_2 specifically (vocab/weights
# are tied to the rebalanced data, not reusable across attempts)
if [ -d tokenizers ]; then
    mv tokenizers models/model_2_rebalanced/
    echo "  moved tokenizers/ -> models/model_2_rebalanced/"
fi
if [ -d checkpoints ]; then
    mv checkpoints models/model_2_rebalanced/
    echo "  moved checkpoints/ -> models/model_2_rebalanced/"
fi

# --- clean up leftover byproduct files ---
echo ""
echo "=== Cleaning up leftover files ==="
# these zips are opus_read's own download cache -- the actual parsed data
# is already in data/raw/<corpus>/corpus.en(.ur), so these are redundant
for zipf in GNOME_latest_moses_en-ur.txt.zip QED_latest_moses_en-ur.txt.zip \
            Tanzil_latest_moses_en-ur.txt.zip Ubuntu_latest_moses_en-ur.txt.zip; do
    if [ -f "$zipf" ]; then
        rm "$zipf"
        echo "  removed redundant $zipf (already parsed into data/raw/)"
    fi
done

if [ -d __pycache__ ]; then
    rm -rf __pycache__
    echo "  removed __pycache__/"
fi

if [ -d test_2 ]; then
    echo "  [NOTE] test_2/ left as-is -- not sure what this is, check it manually"
    echo "         and either move it into models/model_2_rebalanced/ or delete it"
fi

echo ""
echo "=== Writing model READMEs ==="

cat > models/model_1_baseline/README.md << 'EOF'
# Model 1 -- Baseline (OPUS-only)

**Status:** historical record only -- trained weights were not preserved
(checkpoints were cleared before this record was written). Code shown here
reflects what was actually run; re-running it will reproduce comparable
results but not bit-identical ones (data download order/timing can vary).

## Data
OPUS only: OpenSubtitles, Tanzil, QED, GNOME, KDE4, Ubuntu.
~723,697 pairs after cleaning -- but **~97% of this was Tanzil** (Quranic
translation, formal/religious register), since Tanzil alone contributed
748,320 raw pairs versus under 34,000 from every other source combined.

## Model config
- d_model=256, num_layers=4, num_heads=8, d_ff=1024
- 20 epochs, AMP mixed precision, token-count batching (max 4000 tokens/batch)
- Greedy decoding only (no beam search yet)

## Results
- Final train_loss: 1.95, val_loss: 1.72 (in-domain validation split)
- No held-out conversational benchmark was used at this stage
- Qualitative testing revealed the core issue: formal/religious-register
  input translated coherently, but casual conversational input
  (e.g. "Hello, how are you?") caused the decoder to degenerate into a
  repetition loop -- a combination of (a) greedy decoding having no
  mechanism to escape a repetition loop, and (b) the model having seen
  very little everyday conversational Urdu during training

## Key learning -> led to Model 2
The in-domain validation loss looked good, but it was measuring performance
on data drawn from the same (Tanzil-dominated) distribution as training --
it couldn't reveal the domain imbalance problem. This motivated:
1. Adding beam search + n-gram repetition blocking (decoding-level fix)
2. Rebalancing the training data to reduce Tanzil's share and add genuine
   conversational/general-domain sources (OPUS conversational subsets, BPCC)
3. Evaluating against a real out-of-domain conversational benchmark
   (IN22-Conv) instead of trusting in-domain validation loss alone
EOF
echo "  wrote models/model_1_baseline/README.md"

cat > models/model_2_rebalanced/README.md << 'EOF'
# Model 2 -- Rebalanced data + beam search

## Data
Combined and rebalanced to fix Model 1's domain imbalance:

| Source | Pairs | Share |
|---|---|---|
| Tanzil (capped, was ~97%) | 100,000 | 36.5% |
| BPCC-seed (human-curated) | 98,909 | 36.1% |
| BPCC-wiki | 41,335 | 15.1% |
| OpenSubtitles | 29,074 | -- |
| QED | 19,053 | 7.0% |
| GNOME | 11,535 | 4.2% |
| Ubuntu | 3,025 | 1.1% |

~255k pairs after cleaning (dedup, length-ratio filter, max length 80 tokens).

## Model config
Same architecture as Model 1: d_model=256, num_layers=4, num_heads=8,
d_ff=1024, 20 epochs, AMP, token-count batching.

**Decoding: beam search (beam_size=5) with 3-gram repetition blocking**,
replacing Model 1's greedy decoding -- this directly fixes the repetition-
loop failure mode observed in Model 1.

## Results
- Final train_loss: 2.65, val_loss: 2.87 (in-domain validation split --
  higher than Model 1's because the data mix is now genuinely harder/more
  diverse, not because the model is worse)
- **BLEU on IN22-Conv (held-out conversational benchmark, never trained
  on): 22.90**

### Sample translations (IN22-Conv, out-of-domain)
| English | Reference | Model output |
|---|---|---|
| Mom, let's go for a movie tomorrow. | ماں، چلو کل فلم دیکھنے چلتے ہیں۔ | ماں، کل ایک فلم کے لئے جانا۔ |
| I don't have to go to school. | مجھے اسکول نہیں جانا ہے۔ | مجھے اسکول جانے کی ضرورت نہیں ہے۔ |
| It is a holiday. | چھٹی ہے۔ | یہ ایک چھٹی ہے۔ |

## How to reproduce
```bash
pip install -r ../../requirements.txt
python3 download_data.py       # OPUS corpora
python3 download_bpcc.py       # BPCC (requires: hf auth login)
python3 rebalance_data.py      # combine + cap Tanzil + clean
python3 train_tokenizer.py
python3 dataset.py             # smoke test
# then run train_transformer.ipynb top to bottom
```

## Checkpoints
Not committed to this repo (exceeds GitHub's file size limits). Final
checkpoint available at: `<add your Drive/HF link here>`
EOF
echo "  wrote models/model_2_rebalanced/README.md"

echo ""
echo "=== Done ==="
echo "Review the structure with: find models -type f"
echo "Next: add the root README.md and .gitignore (provided separately),"
echo "then: git init && git add . && git commit -m 'Initial commit: model_1 record + model_2 rebalanced'"
