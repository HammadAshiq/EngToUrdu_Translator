"""
Step 2 of the data pipeline: train SentencePiece subword tokenizers.

Separate tokenizers for English and Urdu (not shared) since the two
scripts (Latin vs Perso-Arabic) share essentially no characters --
a shared vocab would just waste embedding capacity on symbols each
language never uses.

Setup:
    pip install sentencepiece
    python3 train_tokenizer.py

Requires data/processed/train.en and data/processed/train.ur to already
exist (produced by download_data.py).
"""

import os
import sentencepiece as spm

VOCAB_SIZE = 16000  # good default for a mid-size corpus; raise toward 32000
                     # if your merged corpus ends up well over ~5M sentence pairs


def train_tokenizer(input_file: str, model_prefix: str, vocab_size: int):
    print(f"Training tokenizer: {model_prefix} (vocab_size={vocab_size})")
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",           # BPE tends to work slightly better than
                                     # unigram for morphologically rich languages
        character_coverage=0.9995,  # close to 1.0 so rare Urdu/Arabic-script
                                     # characters aren't dropped
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        pad_piece="<pad>", unk_piece="<unk>",
        bos_piece="<sos>", eos_piece="<eos>",
    )
    print(f"  wrote {model_prefix}.model and {model_prefix}.vocab")


if __name__ == "__main__":
    os.makedirs("tokenizers", exist_ok=True)

    train_tokenizer(
        input_file="data/processed/train.en",
        model_prefix="tokenizers/en",
        vocab_size=VOCAB_SIZE,
    )
    train_tokenizer(
        input_file="data/processed/train.ur",
        model_prefix="tokenizers/ur",
        vocab_size=VOCAB_SIZE,
    )

    # quick sanity check: encode/decode a sample line from each language
    sp_en = spm.SentencePieceProcessor(model_file="tokenizers/en.model")
    sp_ur = spm.SentencePieceProcessor(model_file="tokenizers/ur.model")

    with open("data/processed/train.en", encoding="utf-8") as f:
        sample_en = f.readline().strip()
    with open("data/processed/train.ur", encoding="utf-8") as f:
        sample_ur = f.readline().strip()

    print("\n--- sanity check ---")
    print(f"EN sample: {sample_en}")
    print(f"  tokens: {sp_en.encode(sample_en, out_type=str)}")
    print(f"  ids:    {sp_en.encode(sample_en, out_type=int)}")
    print(f"  decoded back: {sp_en.decode(sp_en.encode(sample_en, out_type=int))}")

    print(f"\nUR sample: {sample_ur}")
    print(f"  tokens: {sp_ur.encode(sample_ur, out_type=str)}")
    print(f"  ids:    {sp_ur.encode(sample_ur, out_type=int)}")
    print(f"  decoded back: {sp_ur.decode(sp_ur.encode(sample_ur, out_type=int))}")
