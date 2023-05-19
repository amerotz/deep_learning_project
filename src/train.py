import torch
import argparse
from tqdm import tqdm

from lstm_model import *


def main(args):
    vocab_size = 4

    # model, optimizer, loss
    model = LSTMModel(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        embedding_size=args.embedding_size,
        num_layers=args.layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    B = args.batch_size
    L = 5
    batch_num = 100

    x_val = torch.randint(low=0, high=vocab_size, size=(B, L))
    y_val = x_val.flatten()

    for e in range(args.epochs):
        mean_epoch_loss = 0
        for b in tqdm(range(batch_num)):
            # generate mock batch
            batch = torch.randint(low=0, high=vocab_size, size=(B, L))

            # model prediction
            logits = model(batch)
            # get the labels
            target = batch.flatten()

            # compute loss
            loss = loss_fn(input=logits, target=target)
            mean_epoch_loss += loss

            # propagate and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        mean_epoch_loss /= batch_num
        mean_epoch_loss = round(float(mean_epoch_loss), 6)
        print(f"TRAIN EPOCH:{e}/{args.epochs}, LOSS:{mean_epoch_loss}")

        # validation
        with torch.no_grad():
            val_logits = model(x_val)
            validation_loss = loss_fn(input=val_logits, target=y_val)
            validation_loss = round(float(validation_loss), 6)
            print(f"VAL EPOCH:{e}/{args.epochs}, LOSS:{validation_loss}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int, default=1)
    parser.add_argument("-hs", "--hidden_size", type=int, default=16)
    parser.add_argument("-es", "--embedding_size", type=int, default=16)
    parser.add_argument("-l", "--layers", type=int, default=2)
    parser.add_argument("-dp", "--dropout", type=float, default=0.2)
    parser.add_argument("-bi", "--bidirectional", action="store_true", default=False)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.01)
    parser.add_argument("-bs", "--batch_size", type=float, default=100)
    args = parser.parse_args()
    main(args)
