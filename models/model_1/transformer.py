"""
Transformer (Vaswani et al., 2017) built from scratch in PyTorch.

No nn.Transformer, no nn.MultiheadAttention — every matrix multiply,
mask, and normalization is explicit so you can see (and modify) exactly
what's happening at each step.

Designed for a sequence-to-sequence task like English -> Urdu translation:
    Encoder(src) -> memory
    Decoder(tgt, memory) -> logits over target vocab
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Positional Encoding
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (fixed, not learned).

    Since self-attention has no notion of order by itself (it's a weighted
    sum over all positions), we inject position information by adding a
    fixed sinusoidal pattern to the token embeddings.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)  # even dims
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dims
        pe = pe.unsqueeze(0)  # (1, max_len, d_model) -> broadcast over batch

        # register as buffer: moves with .to(device), saved in state_dict,
        # but not a learnable parameter
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# 2. Scaled Dot-Product Multi-Head Attention (fully explicit)
# ---------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    """
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Multi-head: project Q, K, V into `num_heads` smaller subspaces,
    run attention in parallel in each, concatenate, and project back.
    This lets different heads attend to different kinds of relationships
    (e.g. syntactic vs. positional vs. semantic).
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # dimension per head

        # Single big linear layers, then split into heads (equivalent to
        # having a separate projection per head, but one matmul is faster)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)  # output projection

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, num_heads, seq_len, d_k) -> (batch, seq_len, d_model)
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.d_model)

    def forward(self, query, key, value, mask=None):
        """
        query: (batch, q_len, d_model)   -- e.g. decoder hidden states
        key:   (batch, k_len, d_model)   -- e.g. encoder output (cross-attn)
                                             or same as query (self-attn)
        value: (batch, k_len, d_model)
        mask:  (batch, 1, q_len, k_len) or broadcastable, 1 = attend, 0 = block
        """
        Q = self.split_heads(self.w_q(query))  # (b, h, q_len, d_k)
        K = self.split_heads(self.w_k(key))    # (b, h, k_len, d_k)
        V = self.split_heads(self.w_v(value))  # (b, h, k_len, d_k)

        # scaled dot-product attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores: (b, h, q_len, k_len)

        if mask is not None:
            # positions where mask == 0 get -inf so softmax -> ~0 probability
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)  # (b, h, q_len, d_k)
        context = self.combine_heads(context)    # (b, q_len, d_model)

        return self.w_o(context), attn_weights


# ---------------------------------------------------------------------------
# 3. Position-wise Feed-Forward Network
# ---------------------------------------------------------------------------
class PositionwiseFeedForward(nn.Module):
    """Two linear layers with ReLU in between, applied independently to
    every position. This is where most of the "reasoning" capacity of
    each layer lives (attention moves information between positions;
    the FFN transforms it at each position)."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ---------------------------------------------------------------------------
# 4. Encoder Layer
# ---------------------------------------------------------------------------
class EncoderLayer(nn.Module):
    """
    x -> self-attention -> add & norm -> feed-forward -> add & norm
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        # self-attention sublayer (residual + post-norm, as in the original paper)
        attn_out, _ = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout1(attn_out))

        # feed-forward sublayer
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff,
                 max_len=5000, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask):
        # embed + scale (as in the original paper, so embedding magnitude
        # roughly matches the positional encoding magnitude)
        x = self.token_embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x, src_mask)

        return self.norm(x)


# ---------------------------------------------------------------------------
# 5. Decoder Layer
# ---------------------------------------------------------------------------
class DecoderLayer(nn.Module):
    """
    x -> masked self-attention -> add&norm
      -> cross-attention over encoder memory -> add&norm
      -> feed-forward -> add&norm
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask, src_mask):
        # masked self-attention: a position can only attend to itself and
        # earlier positions (causal mask), so training matches inference
        self_attn_out, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn_out))

        # cross-attention: queries come from the decoder, keys/values from
        # the encoder output -- this is how target tokens "look at" the
        # source sentence
        cross_attn_out, cross_attn_weights = self.cross_attn(x, memory, memory, src_mask)
        x = self.norm2(x + self.dropout2(cross_attn_out))

        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_out))
        return x, cross_attn_weights


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff,
                 max_len=5000, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, tgt_mask, src_mask):
        x = self.token_embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        attn_weights = None
        for layer in self.layers:
            x, attn_weights = layer(x, memory, tgt_mask, src_mask)

        return self.norm(x), attn_weights


# ---------------------------------------------------------------------------
# 6. Masks
# ---------------------------------------------------------------------------
def make_src_mask(src: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """Padding mask: block attention to <pad> tokens in the source.
    Shape: (batch, 1, 1, src_len) -- broadcasts over heads and query positions.
    """
    mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask  # 1 = real token, 0 = pad


def make_tgt_mask(tgt: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """Combines padding mask with a causal (look-ahead) mask so the decoder
    can't cheat by attending to future tokens during training.
    Shape: (batch, 1, tgt_len, tgt_len)
    """
    batch_size, tgt_len = tgt.shape

    pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)  # (b, 1, 1, tgt_len)

    causal_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, tgt_len, tgt_len)

    return pad_mask & causal_mask  # (b, 1, tgt_len, tgt_len)


# ---------------------------------------------------------------------------
# 7. Full Transformer (Encoder + Decoder + output projection)
# ---------------------------------------------------------------------------
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        max_len: int = 5000,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.pad_idx = pad_idx

        self.encoder = Encoder(src_vocab_size, d_model, num_layers, num_heads,
                                d_ff, max_len, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_layers, num_heads,
                                d_ff, max_len, dropout)
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)

        self._init_parameters()

    def _init_parameters(self):
        # Xavier init for all weight matrices -- important for Transformers,
        # since the default PyTorch init can make early training unstable
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        """
        src: (batch, src_len) token ids
        tgt: (batch, tgt_len) token ids (decoder input, shifted right by caller)
        returns logits: (batch, tgt_len, tgt_vocab_size)
        """
        src_mask = make_src_mask(src, self.pad_idx)
        tgt_mask = make_tgt_mask(tgt, self.pad_idx)

        memory = self.encoder(src, src_mask)
        # in cross-attention, decoder queries attend over encoder keys/values,
        # so the mask that matters there is the *source* padding mask
        dec_out, _ = self.decoder(tgt, memory, tgt_mask, src_mask)

        logits = self.output_proj(dec_out)
        return logits

    @torch.no_grad()
    def greedy_decode(self, src: torch.Tensor, sos_idx: int, eos_idx: int, max_len: int = 100):
        """Simple greedy decoding for inference (no teacher forcing).
        src: (batch, src_len)
        returns: (batch, generated_len) token ids
        """
        self.eval()
        device = src.device
        batch_size = src.size(0)

        src_mask = make_src_mask(src, self.pad_idx)
        memory = self.encoder(src, src_mask)

        # start every sequence with <sos>
        ys = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys, self.pad_idx)
            dec_out, _ = self.decoder(ys, memory, tgt_mask, src_mask)
            logits = self.output_proj(dec_out[:, -1, :])  # only need last position
            next_token = logits.argmax(dim=-1, keepdim=True)  # (batch, 1)
            ys = torch.cat([ys, next_token], dim=1)

            if (next_token.squeeze(-1) == eos_idx).all():
                break

        return ys

    @torch.no_grad()
    def beam_search_decode(self, src: torch.Tensor, sos_idx: int, eos_idx: int,
                            beam_size: int = 5, max_len: int = 100,
                            length_penalty: float = 0.7, no_repeat_ngram_size: int = 3):
        """Beam search decoding for a single sentence (batch size 1).

        Greedy decoding picks the single best token at each step and can get
        stuck repeating itself once it enters a loop, since it has no way to
        reconsider. Beam search keeps `beam_size` candidate sequences alive
        at once and explores several continuations before committing, which
        largely avoids that failure mode.

        length_penalty: scores are divided by (length ** length_penalty) so
        the search doesn't unfairly favor short sequences (which naturally
        have higher joint probability just from having fewer factors).

        no_repeat_ngram_size: blocks any beam from repeating an n-gram it has
        already produced -- a direct, simple fix for exactly the repetition-
        loop failure you'd see with greedy decoding.
        """
        assert src.size(0) == 1, "beam_search_decode expects a single sentence (batch size 1)"
        self.eval()
        device = src.device

        src_mask = make_src_mask(src, self.pad_idx)
        memory = self.encoder(src, src_mask)  # (1, src_len, d_model)

        # each beam: (token_id_list, cumulative_log_prob, finished_flag)
        beams = [([sos_idx], 0.0, False)]

        def has_repeated_ngram(seq, n):
            if len(seq) < 2 * n:
                return False
            last_ngram = tuple(seq[-n:])
            for i in range(len(seq) - n):
                if tuple(seq[i:i + n]) == last_ngram:
                    return True
            return False

        for _ in range(max_len - 1):
            all_candidates = []
            any_active = False

            for tokens, score, finished in beams:
                if finished:
                    all_candidates.append((tokens, score, True))
                    continue
                any_active = True

                ys = torch.tensor([tokens], dtype=torch.long, device=device)
                tgt_mask = make_tgt_mask(ys, self.pad_idx)
                dec_out, _ = self.decoder(ys, memory, tgt_mask, src_mask)
                logits = self.output_proj(dec_out[:, -1, :])  # (1, vocab)
                log_probs = F.log_softmax(logits, dim=-1).squeeze(0)  # (vocab,)

                topk_log_probs, topk_ids = log_probs.topk(beam_size)

                for lp, tok_id in zip(topk_log_probs.tolist(), topk_ids.tolist()):
                    candidate_tokens = tokens + [tok_id]

                    # skip candidates that would create a repeated n-gram
                    if no_repeat_ngram_size and has_repeated_ngram(
                        candidate_tokens, no_repeat_ngram_size
                    ):
                        continue

                    candidate_finished = (tok_id == eos_idx)
                    all_candidates.append((candidate_tokens, score + lp, candidate_finished))

            if not any_active:
                break

            # rank by length-normalized score, keep top beam_size
            def normalized_score(item):
                tokens, score, _ = item
                length = len(tokens)
                return score / (length ** length_penalty)

            all_candidates.sort(key=normalized_score, reverse=True)
            beams = all_candidates[:beam_size]

            if all(finished for _, _, finished in beams):
                break

        beams.sort(key=lambda item: item[1] / (len(item[0]) ** length_penalty), reverse=True)
        best_tokens = beams[0][0]
        return torch.tensor([best_tokens], dtype=torch.long, device=device)