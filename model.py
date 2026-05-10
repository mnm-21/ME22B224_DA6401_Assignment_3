import math
import copy
import os
import json
import gdown
from typing import Optional, Tuple, List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

# Paths for vocab and checkpointing
_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_VOCAB_PATH = os.path.join(_DIR, "src_vocab.json")
TGT_VOCAB_PATH = os.path.join(_DIR, "tgt_vocab.json")
GDRIVE_FILE_ID = "1m_Vou7_VPi0w9Z4TFFf1gXMj53SkrAaJ"
CHECKPOINT_LOCAL = os.path.join(_DIR, "best_checkpoint.pt")


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Standard scaled dot product attention."""
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    attn_w = torch.softmax(scores, dim=-1)
    # Handle NaN for fully masked rows
    attn_w = attn_w.masked_fill(torch.isnan(attn_w), 0.0)
    output = torch.matmul(attn_w, V)
    return output, attn_w


def make_src_mask(src: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    """Mask out padding in the source sequence."""
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    """Combine padding mask and causal mask for the decoder."""
    _, tgt_len = tgt.size()
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)
    causal = torch.triu(
        torch.ones(tgt_len, tgt_len, device=tgt.device, dtype=torch.bool), diagonal=1
    )
    return pad_mask | causal.unsqueeze(0).unsqueeze(0)


class MultiHeadAttention(nn.Module):
    """Multi-head attention implementation."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = query.size(0)
        # Project and split into heads
        Q = self.W_q(query).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)

        _, attn_w = scaled_dot_product_attention(Q, K, V, mask)
        attn_w = self.dropout(attn_w)
        attn_out = torch.matmul(attn_w, V)

        # Merge heads back
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.W_o(attn_out)


class PositionalEncoding(nn.Module):
    """Injects positional information into the sequence."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class PositionwiseFeedForward(nn.Module):
    """Point-wise feed forward network."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    """Single layer of the encoder."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(p=dropout)
        self.drop2 = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.drop1(self.self_attn(x, x, x, src_mask)))
        x = self.norm2(x + self.drop2(self.feed_forward(x)))
        return x


class DecoderLayer(nn.Module):
    """Single layer of the decoder."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(p=dropout)
        self.drop2 = nn.Dropout(p=dropout)
        self.drop3 = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.norm1(x + self.drop1(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.drop2(self.cross_attn(x, memory, memory, src_mask)))
        x = self.norm3(x + self.drop3(self.feed_forward(x)))
        return x


class Encoder(nn.Module):
    """Stack of encoder layers."""

    def __init__(self, layer: EncoderLayer, N: int):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.self_attn.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """Stack of decoder layers."""

    def __init__(self, layer: DecoderLayer, N: int):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.self_attn.d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):
    """
    Main Transformer class. Loads vocab and weights during init to
    be compatible with the autograder.
    """

    def __init__(
        self,
        src_vocab_size: int = 8500,
        tgt_vocab_size: int = 6500,
        d_model: int = 512,
        N: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        pad_idx: int = 1,
        src_vocab_path: str = SRC_VOCAB_PATH,
        tgt_vocab_path: str = TGT_VOCAB_PATH,
        checkpoint_path: str = CHECKPOINT_LOCAL,
        gdrive_file_id: str = GDRIVE_FILE_ID,
        tie_weights: bool = True,
    ):
        if os.path.exists(src_vocab_path):
            with open(src_vocab_path, "r") as f:
                src_vocab = json.load(f)
            src_vocab_size = len(src_vocab)
        else:
            src_vocab = None

        if os.path.exists(tgt_vocab_path):
            with open(tgt_vocab_path, "r") as f:
                tgt_vocab = json.load(f)
            tgt_vocab_size = len(tgt_vocab)
        else:
            tgt_vocab = None

        super().__init__()
        self.config = {
            "src_vocab_size": src_vocab_size,
            "tgt_vocab_size": tgt_vocab_size,
            "d_model": d_model,
            "N": N,
            "num_heads": num_heads,
            "d_ff": d_ff,
            "dropout": dropout,
        }
        self.d_model = d_model
        self.pad_idx = pad_idx

        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(d_model, dropout)
        self.encoder = Encoder(EncoderLayer(d_model, num_heads, d_ff, dropout), N)
        self.decoder = Decoder(DecoderLayer(d_model, num_heads, d_ff, dropout), N)
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)

        # Weight Tying (Paper Section 3.4)
        if tie_weights:
            self.output_proj.weight = self.tgt_embed.weight

        self._init_parameters()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

        # Load spacy tokenizers, downloading models if not already installed
        import spacy

        def _load_spacy(model_name):
            try:
                return spacy.load(model_name)
            except OSError:
                from spacy.cli import download

                download(model_name)
                return spacy.load(model_name)

        self.src_tokenizer = _load_spacy("de_core_news_sm")
        self.tgt_tokenizer = _load_spacy("en_core_web_sm")

        if gdrive_file_id and gdrive_file_id != "gdrive_file_id>":
            if not os.path.exists(checkpoint_path):
                gdown.download(id=gdrive_file_id, output=checkpoint_path, quiet=False)
            if os.path.exists(checkpoint_path):
                ckpt = torch.load(checkpoint_path, map_location="cpu")
                self.load_state_dict(ckpt["model_state_dict"])

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        emb = self.src_embed(src) * math.sqrt(self.d_model)
        return self.encoder(self.pos_enc(emb), src_mask)

    def decode(
        self,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        emb = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        dec = self.decoder(self.pos_enc(emb), memory, src_mask, tgt_mask)
        return self.output_proj(dec)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def beam_decode(
        self,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        beam_size: int = 5,
        max_len: int = 100,
    ) -> List[int]:
        """Fast beam search that works directly on tensors."""
        device = memory.device
        tgt_sos, tgt_eos = self.tgt_vocab.get("<sos>", 2), self.tgt_vocab.get(
            "<eos>", 3
        )

        beams = [([tgt_sos], 0.0)]
        for _ in range(max_len):
            new_beams = []
            for seq, score in beams:
                if seq[-1] == tgt_eos:
                    new_beams.append((seq, score))
                    continue

                tgt_t = torch.tensor([seq], dtype=torch.long, device=device)
                tgt_mask = make_tgt_mask(tgt_t, self.pad_idx)
                logits = self.decode(memory, src_mask, tgt_t, tgt_mask)
                log_probs = F.log_softmax(logits[:, -1, :], dim=-1)

                top_probs, top_ids = log_probs.topk(beam_size)
                for i in range(beam_size):
                    new_beams.append(
                        (seq + [top_ids[0, i].item()], score + top_probs[0, i].item())
                    )

            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
            if all(b[0][-1] == tgt_eos for b in beams):
                break
        return beams[0][0]

    def infer(self, src_sentence: str, beam_size: int = 5) -> str:
        """End-to-end inference from string."""
        self.eval()
        device = next(self.parameters()).device
        tokens = [tok.text.lower() for tok in self.src_tokenizer(src_sentence)]
        unk, sos, eos = (
            self.src_vocab.get("<unk>", 0),
            self.src_vocab.get("<sos>", 2),
            self.src_vocab.get("<eos>", 3),
        )
        src_ids = [sos] + [self.src_vocab.get(t, unk) for t in tokens] + [eos]
        src_t = torch.tensor([src_ids], dtype=torch.long, device=device)
        src_mask = make_src_mask(src_t, self.pad_idx)

        with torch.no_grad():
            memory = self.encode(src_t, src_mask)
            res_ids = self.beam_decode(memory, src_mask, beam_size)

        idx2tok = {v: k for k, v in self.tgt_vocab.items()}
        tgt_sos, tgt_eos = self.tgt_vocab.get("<sos>", 2), self.tgt_vocab.get(
            "<eos>", 3
        )
        words = [
            idx2tok.get(idx, "<unk>")
            for idx in res_ids
            if idx not in (tgt_sos, tgt_eos)
        ]
        return " ".join(words)
