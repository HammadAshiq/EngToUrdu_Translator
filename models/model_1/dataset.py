"""
Step 3 of the data pipeline: Dataset + DataLoader.

Produces (src, tgt_in, tgt_out) tensors matching what transformer.py expects:
    src     -- source (English) token ids, padded
    tgt_in  -- decoder input: <sos> + target tokens (teacher forcing)
    tgt_out -- decoder target: target tokens + <eos> (what loss is computed against)

Batches by total token count rather than fixed sentence count -- this
matters more than it sounds on a 6GB card: a batch of 32 short sentences
and a batch of 32 very long ones use wildly different memory, so a fixed
batch size either wastes VRAM on short batches or OOMs on long ones.
Token-count batching keeps memory use roughly constant per batch.
"""

import random
import torch
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm

PAD_IDX, UNK_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
MAX_LEN = 80  # matches the length filter used in download_data.py's cleaning step


class TranslationDataset(Dataset):
    def __init__(self, src_file: str, tgt_file: str, sp_src: spm.SentencePieceProcessor,
                 sp_tgt: spm.SentencePieceProcessor, max_len: int = MAX_LEN):
        with open(src_file, encoding="utf-8") as f:
            self.src_lines = [line.strip() for line in f if line.strip()]
        with open(tgt_file, encoding="utf-8") as f:
            self.tgt_lines = [line.strip() for line in f if line.strip()]

        assert len(self.src_lines) == len(self.tgt_lines), \
            "source and target files must have the same number of lines"

        self.sp_src = sp_src
        self.sp_tgt = sp_tgt
        self.max_len = max_len

    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, idx):
        src_ids = self.sp_src.encode(self.src_lines[idx], out_type=int)[: self.max_len]

        tgt_ids = self.sp_tgt.encode(self.tgt_lines[idx], out_type=int)[: self.max_len - 1]
        tgt_in = [SOS_IDX] + tgt_ids
        tgt_out = tgt_ids + [EOS_IDX]

        return {
            "src": src_ids,
            "tgt_in": tgt_in,
            "tgt_out": tgt_out,
            "n_tokens": len(src_ids) + len(tgt_in),  # used for token-count batching
        }


def collate_fn(batch):
    """Pads every sequence in the batch to the batch's own max length
    (not a fixed global max) -- keeps padding waste to a minimum."""

    def pad(seqs, pad_value=PAD_IDX):
        max_len = max(len(s) for s in seqs)
        return torch.tensor(
            [s + [pad_value] * (max_len - len(s)) for s in seqs], dtype=torch.long
        )

    src = pad([item["src"] for item in batch])
    tgt_in = pad([item["tgt_in"] for item in batch])
    tgt_out = pad([item["tgt_out"] for item in batch])

    return src, tgt_in, tgt_out


class TokenCountBatchSampler:
    """Groups example indices into batches capped by total token count,
    not example count. Shuffles bucket order each epoch but sorts within
    a shuffled chunk by length first, to keep padding low without fully
    destroying randomness (a standard trick in NMT training)."""

    def __init__(self, lengths, max_tokens: int = 4000, shuffle: bool = True):
        self.lengths = lengths
        self.max_tokens = max_tokens
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            random.shuffle(indices)

        # sort in large chunks by length to reduce padding, while the chunk
        # order itself stays shuffled so training isn't fully sorted by length
        chunk_size = 2000
        chunks = [indices[i:i + chunk_size] for i in range(0, len(indices), chunk_size)]
        for chunk in chunks:
            chunk.sort(key=lambda i: self.lengths[i])

        batches = []
        current_batch, current_tokens = [], 0
        for chunk in chunks:
            for idx in chunk:
                item_tokens = self.lengths[idx]
                if current_batch and current_tokens + item_tokens > self.max_tokens:
                    batches.append(current_batch)
                    current_batch, current_tokens = [], 0
                current_batch.append(idx)
                current_tokens += item_tokens
            if current_batch:
                batches.append(current_batch)
                current_batch, current_tokens = [], 0

        if self.shuffle:
            random.shuffle(batches)

        yield from batches

    def __len__(self):
        # approximate -- exact count depends on shuffled bucketing
        total_tokens = sum(self.lengths)
        return max(1, total_tokens // self.max_tokens)


def build_dataloader(src_file, tgt_file, sp_src, sp_tgt, max_tokens=4000, shuffle=True):
    dataset = TranslationDataset(src_file, tgt_file, sp_src, sp_tgt)
    lengths = [dataset[i]["n_tokens"] for i in range(len(dataset))]
    sampler = TokenCountBatchSampler(lengths, max_tokens=max_tokens, shuffle=shuffle)

    return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn)


if __name__ == "__main__":
    # quick smoke test
    sp_en = spm.SentencePieceProcessor(model_file="tokenizers/en.model")
    sp_ur = spm.SentencePieceProcessor(model_file="tokenizers/ur.model")

    loader = build_dataloader(
        "data/processed/train.en", "data/processed/train.ur",
        sp_en, sp_ur, max_tokens=2000,
    )

    print(f"~{len(loader)} batches per epoch (approx, token-count batching)")
    for i, (src, tgt_in, tgt_out) in enumerate(loader):
        print(f"batch {i}: src {tuple(src.shape)}, tgt_in {tuple(tgt_in.shape)}, "
              f"tgt_out {tuple(tgt_out.shape)}")
        if i >= 4:
            break
