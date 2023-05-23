import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        embedding_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm_layer = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )

        self.multiplier = 2 if bidirectional else 1
        self.prediction_layer = nn.Linear(self.multiplier * hidden_size, vocab_size)
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
        lstm_out, _ = self.lstm_layer(embedding)

        # (B, L, H) -> (B, L, V)
        logits = self.prediction_layer(lstm_out)

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
