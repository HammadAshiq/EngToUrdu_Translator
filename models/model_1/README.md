# Model 1 -- English to Urdu Transformer (from scratch)

A Transformer (Vaswani et al., 2017) built from scratch in PyTorch --
no `nn.Transformer`, no `nn.MultiheadAttention`.

## Data

Combined from multiple sources, with domain balance considered explicitly
rather than just taking whatever was largest:

| Source | Pairs | Share |
|---|---|---|
| Tanzil (capped) | 100,000 | 36.5% |
| BPCC-seed (human-curated) | 98,909 | 36.1% |
| BPCC-wiki | 41,335 | 15.1% |
| OpenSubtitles | 29,074 | -- |
| QED | 19,053 | 7.0% |
| GNOME | 11,535 | 4.2% |
| Ubuntu | 3,025 | 1.1% |

~255k pairs after cleaning (dedup, length-ratio filter, max length 80 tokens).

Tanzil (Quranic translation -- formal/religious register) was deliberately
capped rather than used in full: an earlier uncapped run made up ~97% of
the training data, which produced a model that translated formal register
well but degenerated on everyday conversational input. Capping it and
adding genuinely conversational/general-domain sources fixed this.

## Model config

- d_model=256, num_layers=4, num_heads=8, d_ff=1024
- 20 epochs, AMP mixed precision, token-count batching (max 4000 tokens/batch)
- LR warmup + inverse-sqrt decay schedule (as in the original paper)
- **Decoding: beam search (beam_size=5) with 3-gram repetition blocking**

## Results

- Final train_loss: 2.65, val_loss: 2.87 (in-domain validation split)
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
