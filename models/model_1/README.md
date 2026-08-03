# Model 1 -- English to Urdu Transformer (from scratch)

A Transformer (Vaswani et al., 2017) built from scratch in PyTorch -- no
`nn.Transformer`, no `nn.MultiheadAttention`. Trained in two stages: an
initial rebalanced-data run, then a significantly larger data scale-up.

## Architecture

- d_model=256, num_layers=4, num_heads=8, d_ff=1024
- AMP mixed precision, token-count batching, gradient accumulation
- LR warmup + inverse-sqrt decay (as in the original paper)
- **Beam search decoding (beam_size=5) with 3-gram repetition blocking**
  -- fixes a repetition-loop failure mode that greedy decoding had on
  casual/conversational input

## Data -- two stages

**Stage 1 (255k pairs):** OPUS (OpenSubtitles, QED, GNOME, Ubuntu) + a
first pass of BPCC (seed + wiki subsets), with Tanzil (Quranic
translation -- formal/religious register) deliberately capped at 100k.
An earlier *uncapped* attempt let Tanzil make up ~97% of the training
data, which produced a model that translated formal register well but
degenerated into repetition loops on everyday conversational input --
capping it and adding genuinely conversational sources fixed this.

**Stage 2 (1.84M pairs, ~7x larger):** added BPCC's larger subsets --
`daily` (human-curated everyday conversation), `ilci`, `massive`,
`samanantar_v2` (485k, kept in full), and `nllb_filtered` (capped at 1M
from a raw 5.3M -- otherwise it alone would have dominated the corpus
the same way uncapped Tanzil did in the first attempt). Tokenizer vocab
raised from 16k to 32k to match the larger, more diverse corpus.

| Source | Pairs | Share |
|---|---|---|
| BPCC-nllb_filtered (capped) | 1,000,000 | 52.3% |
| BPCC-samanantar_v2 | 484,920 | 25.3% |
| BPCC-ilci | 100,967 | 5.3% |
| Tanzil (capped) | 100,000 | 5.2% |
| BPCC-seed | 98,909 | 5.2% |
| BPCC-wiki | 41,335 | 2.2% |
| OpenSubtitles | 29,074 | 1.5% |
| QED | 19,053 | 1.0% |
| BPCC-massive | 16,490 | 0.9% |
| GNOME | 11,535 | 0.6% |
| BPCC-daily | 8,444 | 0.4% |
| Ubuntu | 3,025 | 0.2% |

~1.84M pairs after cleaning (dedup, length-ratio filter, max length 80 tokens).

## Results

| Stage | Data | Epochs | Val loss | BLEU (IN22-Conv) |
|---|---|---|---|---|
| 1 | 255k pairs | 20 | 2.87 (converged) | 22.90 |
| 2 | 1.84M pairs | 12 | 3.18 (**still decreasing**) | 24.14 |

BLEU is measured on IN22-Conv (AI4Bharat), a 1,503-sentence conversational
benchmark held out entirely from training.

**Note on Stage 2 convergence:** the 7x larger corpus needs more than 12
epochs to converge -- val loss was still dropping at a meaningful rate
when training stopped (unlike Stage 1, which had genuinely flattened by
epoch 18-20). The BLEU gain from Stage 1 to Stage 2 (+1.24) is therefore
likely an *underestimate* of what this data can deliver; resuming
training from `checkpoints/latest.pt` for more epochs is the most direct
way to improve further, ahead of adding yet more data or a larger model.

### Sample translations (IN22-Conv, out-of-domain)
| English | Reference | Model output |
|---|---|---|
| Mom, let's go for a movie tomorrow. | ماں، چلو کل فلم دیکھنے چلتے ہیں۔ | ماں، چلو کل ایک فلم کے لئے جاؤ۔ |
| I don't have to go to school. | مجھے اسکول نہیں جانا ہے۔ | مجھے اسکول جانے کی ضرورت نہیں ہے۔ |
| It is a holiday. | چھٹی ہے۔ | یہ چھٹی ہے۔ |
| Oh, tomorrow is the 14th of April right? | اوہ، کل اپریل کی 14 ہے نا؟ | کیا کل 14 اپریل کا دن ہے؟ |

## How to use the trained model

**Command line / scripting** (`translate.py` -- resolves its own paths,
works from any directory):
```bash
python3 translate.py "Hello, how are you?"
python3 translate.py --interactive
python3 translate.py --file sentences.txt --output translations.txt
```

**Web interface** (`app.py`, Gradio):
```bash
pip install gradio
python3 app.py
```
Opens a local web UI with example sentences and a beam-size slider.

## How to reproduce training

```bash
pip install -r ../../requirements.txt

# from the project root:
python3 models/model_1/download_data.py     # OPUS corpora
python3 models/model_1/download_bpcc.py      # BPCC (requires: hf auth login)
python3 models/model_1/rebalance_data.py     # combine + cap + clean
python3 models/model_1/train_tokenizer.py
python3 models/model_1/dataset.py            # smoke test
# then open train_transformer.ipynb from the project root and run top to bottom
```

## Checkpoints

Not committed to this repo (exceeds GitHub's file size limits). Final
checkpoint available at: `<add your Drive/HF link here>`