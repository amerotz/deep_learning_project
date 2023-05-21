import torch
import torch.nn as nn


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
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_size,
            nhead=attention_heads,
            dim_feedforward=hidden_size,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=self.decoder_layer,
            num_layers=num_layers,
        )

        self.prediction_layer = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    # batch (B, L) -> (B * L, V)
    def forward(self, batch):
        B = batch.size(0)
        L = batch.size(1)
        H = self.hidden_size
        V = self.vocab_size

        # (B, L) -> (B, L, E)
        embedding = self.embedding(batch)

        # (B, L, E) -> (B, L, H)
        transf_out = self.transformer_decoder(embedding)

        # (B, L, H) -> (B * L, H)
        pred_input = transf_out.reshape(B * L, H)

        # (B * L, H) -> (B * L, V)
        logits = self.prediction_layer(pred_input)

        # (B * L, V) -> (B, L, V)
        # logits = logits.reshape(B, L, V)

        return logits

    def inference(
        self, sos_idx, eos_idx, max_len=512, mode="greedy", device="cpu", temperature=1
    ):
        with torch.no_grad():
            generation = [sos_idx]
            t = 0
            while t < max_len:
                input = torch.LongTensor([generation[-1]]).to(device).unsqueeze(0)

                out = self.forward(input)

                tok = self.sample(out, mode=mode)
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
