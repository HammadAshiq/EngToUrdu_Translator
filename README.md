# English to Urdu Translator

A Transformer (Vaswani et al., 2017) built from scratch in PyTorch --
no `nn.Transformer`, no `nn.MultiheadAttention` -- trained for English to
Urdu translation.

**Current result: BLEU 24.14** on IN22-Conv, a held-out conversational
benchmark, from a model trained on ~1.84M sentence pairs. See
[`models/model_1/README.md`](models/model_1/README.md) for the full
data pipeline, architecture, and results breakdown.

## Try it

```bash
pip install -r requirements.txt

# command line
python3 models/model_1/translate.py "Hello, how are you?"

# web interface
python3 models/model_1/app.py
```

(Requires a trained checkpoint at `models/model_1/checkpoints/latest.pt`
-- see the model README for training instructions, or download a
pretrained checkpoint from the link in that README.)

## Project structure

Each significant model iteration lives in its own self-contained folder
under `models/`, with the exact code, config, and results for that attempt.

- [`models/model_1/`](models/model_1/README.md) -- rebalanced training
  data, then scaled up ~7x with additional mined/curated sources, plus
  beam search decoding and a standalone inference script + web UI.

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