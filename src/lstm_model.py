import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils


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

        multiplier = 2 if bidirectional else 1
        self.prediction_layer = nn.Linear(multiplier * hidden_size, vocab_size)

    # batch (B, L, V)
    def forward(self, batch):
        embedding = self.embedding(batch)
        pass
