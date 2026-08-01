"""
Sanity checks for transformer.py:
1. Forward pass shapes are correct.
2. Masks actually block what they should (causal + padding).
3. The model can overfit a tiny dummy dataset (loss -> ~0), which is the
   standard way to confirm there's no wiring bug before you touch real data.
"""

import torch
import torch.nn as nn

from transformer import Transformer, make_src_mask, make_tgt_mask

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2

torch.manual_seed(0)


def test_shapes():
    src_vocab_size = 50
    tgt_vocab_size = 60
    batch_size = 4
    src_len = 10
    tgt_len = 8

    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=64,
        num_layers=2,
        num_heads=4,
        d_ff=128,
        pad_idx=PAD_IDX,
    )

    src = torch.randint(3, src_vocab_size, (batch_size, src_len))
    tgt = torch.randint(3, tgt_vocab_size, (batch_size, tgt_len))

    logits = model(src, tgt)
    assert logits.shape == (batch_size, tgt_len, tgt_vocab_size), logits.shape
    print(f"[OK] forward pass shape: {logits.shape}")


def test_causal_mask_blocks_future():
    tgt = torch.tensor([[1, 5, 6, 7]])  # (1, 4), no padding
    mask = make_tgt_mask(tgt, PAD_IDX)  # (1, 1, 4, 4)
    mask = mask[0, 0]

    # position i should NOT be able to attend to position j > i
    for i in range(4):
        for j in range(4):
            allowed = mask[i, j].item()
            expected = j <= i
            assert allowed == expected, f"mask[{i},{j}]={allowed}, expected {expected}"
    print("[OK] causal mask correctly blocks future positions")


def test_padding_mask_blocks_pad():
    src = torch.tensor([[5, 6, 7, PAD_IDX, PAD_IDX]])  # last two are padding
    mask = make_src_mask(src, PAD_IDX)  # (1, 1, 1, 5)
    mask = mask[0, 0, 0]
    assert mask.tolist() == [True, True, True, False, False]
    print("[OK] padding mask correctly blocks <pad> tokens")


def test_overfit_tiny_batch():
    """If the model can't drive loss near zero on 2 fixed sentences with
    enough steps, something in the wiring is broken."""
    src_vocab_size = 20
    tgt_vocab_size = 20

    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=64,
        num_layers=2,
        num_heads=4,
        d_ff=128,
        pad_idx=PAD_IDX,
    )

    # two toy "sentences" (already tokenized as ints), padded to equal length
    src = torch.tensor([
        [4, 5, 6, 7, PAD_IDX],
        [8, 9, 10, 11, 12],
    ])
    # decoder input = <sos> + sentence (teacher forcing)
    tgt_in = torch.tensor([
        [SOS_IDX, 4, 5, 6, 7],
        [SOS_IDX, 8, 9, 10, 11],
    ])
    # decoder target = sentence + <eos> (shifted by one vs. tgt_in)
    tgt_out = torch.tensor([
        [4, 5, 6, 7, EOS_IDX],
        [8, 9, 10, 11, EOS_IDX],
    ])

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    model.train()
    for step in range(300):
        optimizer.zero_grad()
        logits = model(src, tgt_in)  # (batch, tgt_len, vocab)
        loss = criterion(logits.reshape(-1, tgt_vocab_size), tgt_out.reshape(-1))
        loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == 299:
            print(f"  step {step:3d}  loss {loss.item():.4f}")

    assert loss.item() < 0.1, f"Model failed to overfit tiny batch, final loss={loss.item()}"
    print(f"[OK] model overfits tiny batch (final loss={loss.item():.4f})")


def test_greedy_decode_runs():
    model = Transformer(
        src_vocab_size=20, tgt_vocab_size=20,
        d_model=32, num_layers=1, num_heads=2, d_ff=64, pad_idx=PAD_IDX,
    )
    src = torch.tensor([[4, 5, 6, 7, PAD_IDX]])
    out = model.greedy_decode(src, sos_idx=SOS_IDX, eos_idx=EOS_IDX, max_len=10)
    print(f"[OK] greedy_decode runs, output shape {out.shape}: {out.tolist()}")


if __name__ == "__main__":
    print("Running transformer sanity checks...\n")
    test_shapes()
    test_causal_mask_blocks_future()
    test_padding_mask_blocks_pad()
    print("\nOverfitting tiny batch (should reach near-zero loss):")
    test_overfit_tiny_batch()
    print()
    test_greedy_decode_runs()
    print("\nAll checks passed.")
