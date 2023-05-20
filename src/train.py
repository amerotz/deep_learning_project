import torch
import torch.nn as nn
import torch.utils.data as tud
import argparse
from tqdm import tqdm

from lstm_model import *
from dataset import *


def main(args):
    print("Loading data...")
    # load dataset
    dataset = MusicDataset(
        data_file="data/dataset.json",
        vocab_file="data/vocab.json",
        max_sequence_length=512,
        create_data=False,
    )

    vocab_size = dataset.vocab_size
    B = args.batch_size
    L = dataset.max_sequence_length

    # split data
    print("Creating splits...")
    train_data, val_data = tud.random_split(
        dataset, [args.train_ratio, 1 - args.train_ratio]
    )

    # wrap in data loaders
    train_loader = tud.DataLoader(train_data, batch_size=B, shuffle=True)
    val_loader = tud.DataLoader(val_data, batch_size=len(val_data), shuffle=True)

    print("Creating model...")
    # model, optimizer, loss
    model = LSTMModel(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        embedding_size=args.embedding_size,
        num_layers=args.layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
    )
    print(model)

    # check gpu
    device = "cpu"
    if torch.cuda.is_available():
        print("Using CUDA.")
        model = model.cuda()
        device = "cuda"

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss(ignore_index=dataset.pad_idx)

    print("Training started.")
    for e in range(args.epochs):
        mean_epoch_loss = 0
        batch_num = 0

        for input, target in iter(train_loader):
            # model prediction
            logits = model(input.to(device))

            # get the labels
            target = target.flatten().to(device)

            # compute loss
            loss = loss_fn(input=logits, target=target)
            mean_epoch_loss += loss

            # propagate and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_num += 1

        mean_epoch_loss /= batch_num
        mean_epoch_loss = round(float(mean_epoch_loss), 6)
        print(f"TRAIN\tEPOCH:{e}/{args.epochs}\tLOSS:{mean_epoch_loss}")

        # validation
        with torch.no_grad():
            # get val data
            x_val, y_val = list(iter(val_loader))[0]
            # forward
            val_logits = model(x_val.to(device))
            # loss
            validation_loss = loss_fn(
                input=val_logits, target=y_val.flatten().to(device)
            )
            validation_loss = round(float(validation_loss), 6)
            print(f"VAL\tEPOCH:{e}/{args.epochs}\tLOSS:{validation_loss}")

    gen = model.inference(dataset.sos_idx, dataset.eos_idx, device=device)
    gen = [dataset.i2w[str(i)] for i in gen]
    print(" ".join(gen))


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
    parser.add_argument("-tr", "--train_ratio", type=float, default=0.8)
    args = parser.parse_args()
    main(args)
