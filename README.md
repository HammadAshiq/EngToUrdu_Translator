# English to Urdu Translator

A Transformer (Vaswani et al., 2017) built from scratch in PyTorch --
no `nn.Transformer`, no `nn.MultiheadAttention` -- trained for English to
Urdu translation.

## Project structure

Each significant model iteration lives in its own self-contained folder
under `models/`, with the exact code, config, and results for that attempt.
Later folders build on earlier ones' findings -- see each folder's README
for what changed and why.

- [`models/model_1/`](models/model_1/README.md) -- rebalanced training
  data + beam search decoding. BLEU 22.90 on a held-out conversational
  benchmark (IN22-Conv).

Future improvements will be added as `model_2/`, `model_3/`, etc.,
following the same pattern.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Each model folder's README has the exact commands to reproduce that
specific run (data download, tokenizer training, model training).

## Requirements

- Python 3.12
- An NVIDIA GPU is strongly recommended (training was done on an RTX 3050,
  6GB VRAM, with AMP mixed precision) -- CPU training is possible but
  dramatically slower
