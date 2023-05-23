import torch
import math
import torch.nn as nn


# from pytorch docs
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x.swapaxes(0, 1)
        x = x + self.pe[: x.size(0)]
        x = self.dropout(x)
        x = x.swapaxes(0, 1)
        return x


class TransfModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        embedding_size: int,
        num_layers: int,
        dropout: float,
        attention_heads: int,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.attention_heads = attention_heads

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=attention_heads,
            dim_feedforward=hidden_size,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=num_layers,
        )

        self.positional_encoding = PositionalEncoding(
            d_model=embedding_size, dropout=dropout, max_len=512
        )

        self.prediction_layer = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    # batch (B, L) -> (B * L, V)
    def forward(self, batch):
        B = batch.size(0)
        L = batch.size(1)
        H = self.hidden_size
        V = self.vocab_size

        mask = nn.Transformer.generate_square_subsequent_mask(
            L, device=batch.device
        ).bool()

        # (B, L) -> (B, L, E)
        embedding = self.embedding(batch)

        pos_embedding = self.positional_encoding(embedding)

        # (B, L, E) -> (B, L, H)
        transf_out = self.transformer_encoder(pos_embedding, mask=mask)

        # (B, L, H) -> (B, L, V)
        logits = self.prediction_layer(transf_out)

        return logits

    def inference(
        self, sos_idx, eos_idx, max_len=512, mode="greedy", device="cpu", temperature=1
    ):
        with torch.no_grad():
            generation = [sos_idx]
            t = 0
            while t < max_len:
                input = torch.LongTensor(generation).to(device).unsqueeze(0)

                out = self.forward(input)[:, -1, :]

                tok = self.sample(out, mode=mode, T=temperature)
                generation.append(tok)

                if tok == eos_idx:
                    break

                t += 1

            return generation

    def sample(self, out, mode="greedy", K=5, T=1, P=0.9):
        if mode == "greedy":
            sample = torch.argmax(out)

        elif mode == "topk":
            values, indexes = torch.topk(out, K, dim=-1)
            out = out.clone().squeeze(1)
            out[out < values[:, -1]] = -float("Inf")
            probs = self.softmax(out / T).squeeze()
            sample = torch.multinomial(probs, 1)

        elif mode == "topp":
            values, indexes = torch.sort(out / T, descending=True)
            values = self.softmax(values)
            cum_probs = torch.cumsum(values, dim=-1)

            remove = cum_probs > P
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = 0

            out = out.clone()
            remove = torch.zeros_like(out, dtype=torch.bool).scatter_(
                dim=-1, index=indexes, src=remove
            )
            out[remove] = -float("Inf")

            probs = self.softmax(out / T).squeeze()
            sample = torch.multinomial(probs, 1)

        return sample.item()
