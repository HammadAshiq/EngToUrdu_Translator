#!/usr/bin/env python3
"""
Standalone inference for the trained EN -> Urdu Transformer.

Unlike the training scripts in this project, this one resolves its own
paths relative to its own file location (not the current working
directory) -- so it works correctly no matter where you run it from,
no need to cd to the project root first.

Usage:
    python3 translate.py "Hello, how are you?"
    python3 translate.py --interactive
    python3 translate.py --file sentences.txt --output translations.txt
"""

import argparse
import sys
from pathlib import Path

import torch
import sentencepiece as spm

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))  # so `import transformer` finds transformer.py here

from transformer import Transformer

PAD_IDX, UNK_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3

DEFAULT_CHECKPOINT = THIS_DIR / "checkpoints" / "latest.pt"
DEFAULT_TOKENIZER_EN = THIS_DIR / "tokenizers" / "en.model"
DEFAULT_TOKENIZER_UR = THIS_DIR / "tokenizers" / "ur.model"


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]

    if not DEFAULT_TOKENIZER_EN.exists() or not DEFAULT_TOKENIZER_UR.exists():
        print(f"Error: tokenizers not found at {DEFAULT_TOKENIZER_EN} / {DEFAULT_TOKENIZER_UR}",
              file=sys.stderr)
        sys.exit(1)

    sp_en = spm.SentencePieceProcessor(model_file=str(DEFAULT_TOKENIZER_EN))
    sp_ur = spm.SentencePieceProcessor(model_file=str(DEFAULT_TOKENIZER_UR))

    model = Transformer(
        src_vocab_size=sp_en.get_piece_size(),
        tgt_vocab_size=sp_ur.get_piece_size(),
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        max_len=config["max_len"],
        dropout=config["dropout"],
        pad_idx=PAD_IDX,
    ).to(device)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print(f"Loaded checkpoint: step {checkpoint['step']}, epoch {checkpoint['epoch']}", file=sys.stderr)
    return model, sp_en, sp_ur


def translate(model, sp_en, sp_ur, device, sentence: str, beam_size: int = 5, max_len: int = 80) -> str:
    src_ids = sp_en.encode(sentence, out_type=int)
    src = torch.tensor([src_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        out_ids = model.beam_search_decode(
            src, sos_idx=SOS_IDX, eos_idx=EOS_IDX,
            beam_size=beam_size, max_len=max_len, no_repeat_ngram_size=3,
        )
    out_ids = out_ids[0].tolist()

    if EOS_IDX in out_ids:
        out_ids = out_ids[1:out_ids.index(EOS_IDX)]
    else:
        out_ids = out_ids[1:]

    return sp_ur.decode(out_ids)


def main():
    parser = argparse.ArgumentParser(description="Translate English text to Urdu")
    parser.add_argument("text", nargs="?", help="English sentence to translate")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT),
                         help="Path to model checkpoint (default: checkpoints/latest.pt)")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--interactive", action="store_true", help="Interactive REPL mode")
    parser.add_argument("--file", help="Translate each line of this file")
    parser.add_argument("--output", help="Write translations to this file (used with --file)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", file=sys.stderr)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: checkpoint not found at {checkpoint_path}", file=sys.stderr)
        print("Train a model first (see train_transformer.ipynb), or pass "
              "--checkpoint pointing at a valid .pt file.", file=sys.stderr)
        sys.exit(1)

    model, sp_en, sp_ur = load_model(checkpoint_path, device)

    if args.file:
        with open(args.file, encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        results = []
        for i, line in enumerate(lines):
            out = translate(model, sp_en, sp_ur, device, line, beam_size=args.beam_size)
            results.append(out)
            print(f"[{i + 1}/{len(lines)}] {line} -> {out}", file=sys.stderr)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write("\n".join(results))
            print(f"Wrote {len(results)} translations to {args.output}", file=sys.stderr)
        else:
            for r in results:
                print(r)

    elif args.interactive:
        print("Interactive mode -- type English text, Ctrl+C or 'quit' to exit.", file=sys.stderr)
        while True:
            try:
                text = input("EN> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if text.lower() in ("quit", "exit"):
                break
            if not text:
                continue
            print(f"UR> {translate(model, sp_en, sp_ur, device, text, beam_size=args.beam_size)}")

    elif args.text:
        print(translate(model, sp_en, sp_ur, device, args.text, beam_size=args.beam_size))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()