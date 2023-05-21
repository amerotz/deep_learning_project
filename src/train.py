import torch
import torch.nn as nn
import torch.utils.data as tud
import argparse
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from lstm_model import *
from transf_model import *
from dataset import *


def pprint(s):
    s = str(s)
    pprint.log += s + "\n"
    print(s)


pprint.log = ""


def main(args):
    pprint("Loading data...")

    # load dataset
    if args.create_data:
        dataset = MusicDataset(
            data_file="data/dataset.txt",
            max_sequence_length=args.max_sequence_length,
            create_data=True,
        )

    else:
        dataset = MusicDataset(
            data_file="data/dataset.json",
            vocab_file="data/vocab.json",
            max_sequence_length=args.max_sequence_length,
            create_data=False,
        )

    vocab_size = dataset.vocab_size
    B = args.batch_size
    L = dataset.max_sequence_length

    if not args.inference:
        # split data
        pprint("Creating splits...")
        train_data, val_data = tud.random_split(
            dataset, [args.train_ratio, 1 - args.train_ratio]
        )

        # wrap in data loaders
        train_loader = tud.DataLoader(train_data, batch_size=B, shuffle=True)
        val_loader = tud.DataLoader(val_data, batch_size=len(val_data), shuffle=True)

    pprint("Creating model...")

    # model, optimizer, loss
    if args.architecture == "lstm":
        model = LSTMModel(
            vocab_size=vocab_size,
            hidden_size=args.hidden_size,
            embedding_size=args.embedding_size,
            num_layers=args.layers,
            dropout=args.dropout,
            bidirectional=False,
        )
    elif args.architecture == "transf":
        model = TransfModel(
            vocab_size=vocab_size,
            hidden_size=args.hidden_size,
            embedding_size=args.embedding_size,
            num_layers=args.layers,
            dropout=args.dropout,
            attention_heads=args.attention_heads,
        )
    pprint(model)

    # to keep track of epochs across multiple runs
    offset = args.epochs_offset + 1
    # load previous checkpoint
    if args.load != None:
        pprint(f"Loading checkpoint {args.load}")
        model.load_state_dict(torch.load(args.load))

    # check gpu
    device = "cpu"
    if torch.cuda.is_available():
        pprint("Using CUDA.")
        model = model.cuda()
        device = "cuda"

    # optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss(ignore_index=dataset.pad_idx)

    # early stopping
    patience = args.patience
    old_validation_loss = float("Inf")

    # just training
    if not args.inference:
        epoch_training_loss = []
        epoch_validation_loss = []

        os.makedirs(args.ckpt_dir, exist_ok=True)

        pprint("Training started.")
        for e in range(args.epochs):
            mean_epoch_loss = 0
            batch_num = 0

            model.train()
            for input, target in iter(train_loader):
                # get the labels
                target = target.to(device)

                # model prediction
                logits = model(input.to(device))

                # compute loss
                loss = loss_fn(input=logits, target=target.flatten())
                mean_epoch_loss += loss

                # propagate and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_num += 1

            mean_epoch_loss /= batch_num
            mean_epoch_loss = float(mean_epoch_loss)
            # log
            epoch_training_loss.append(mean_epoch_loss)
            mean_epoch_loss = round(mean_epoch_loss, 6)
            pprint(f"TRAIN\tEPOCH:{e}/{args.epochs}\tLOSS:{mean_epoch_loss}")

            # validation
            model.eval()
            with torch.no_grad():
                # get val data
                x_val, y_val = list(iter(val_loader))[0]
                y_val = y_val.to(device)
                # forward
                val_logits = model(x_val.to(device))
                # loss
                validation_loss = loss_fn(input=val_logits, target=y_val.flatten())
                validation_loss = float(validation_loss)
                # log
                epoch_validation_loss.append(validation_loss)
                validation_loss = round(validation_loss, 6)
                pprint(f"VAL\tEPOCH:{e}/{args.epochs}\tLOSS:{validation_loss}")

                if validation_loss > old_validation_loss:
                    patience -= 1
                else:
                    patience = args.patience

                old_validation_loss = validation_loss

            checkpoint_path = f"{args.ckpt_dir}/E{e + offset}.pytorch"
            if e % 5 == 0:
                torch.save(model.state_dict(), checkpoint_path)
                pprint("Model saved at %s" % checkpoint_path)

            if patience == 0:
                pprint("Patience reached. Early stopping.")
                torch.save(model.state_dict(), checkpoint_path)
                pprint("Model saved at %s" % checkpoint_path)
                break

    model_name = f"{args.architecture}_l={args.layers}_es={args.embedding_size}_hs={args.hidden_size}_d={args.dropout}_e={args.epochs}_lr={args.learning_rate}_bs={args.batch_size}"
    if not args.inference:
        plt.plot(epoch_training_loss, label="training loss")
        plt.plot(epoch_validation_loss, label="validation loss")
        plt.yscale("log")
        plt.legend()
        plt.xticks(range(0, args.epochs, max(1, args.epochs // 20)))
        plt.grid()
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.savefig(f"{args.ckpt_dir}/{model_name}.png")

    # inference at end of training or because args.inference
    model.eval()
    pprint("Inferenced sample")
    gen = model.inference(
        dataset.sos_idx,
        dataset.eos_idx,
        device=device,
        mode=args.mode,
        temperature=args.temperature,
    )
    gen = [dataset.i2w[str(i)] for i in gen]
    pprint(" ".join(gen))

    with open(f"{args.ckpt_dir}/{model_name}.log", "w") as f:
        f.write(pprint.log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int, default=1)
    parser.add_argument("-hs", "--hidden_size", type=int, default=16)
    parser.add_argument("-es", "--embedding_size", type=int, default=16)
    parser.add_argument("-l", "--layers", type=int, default=2)
    parser.add_argument("-dp", "--dropout", type=float, default=0.2)
    parser.add_argument("-ah", "--attention_heads", type=int, default=2)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.01)
    parser.add_argument("-bs", "--batch_size", type=float, default=100)
    parser.add_argument("-tr", "--train_ratio", type=float, default=0.9)
    parser.add_argument("-cd", "--create_data", action="store_true")
    parser.add_argument("-p", "--patience", type=int, default=5)
    parser.add_argument("-ld", "--load", type=str, default=None)
    parser.add_argument("-i", "--inference", action="store_true")
    parser.add_argument("-m", "--mode", type=str, default="greedy")
    parser.add_argument("-t", "--temperature", type=float, default=1)
    parser.add_argument("-ml", "--max_sequence_length", type=int, default=512)
    parser.add_argument("-ckd", "--ckpt_dir", type=str, default="./ckpts")
    parser.add_argument("-arch", "--architecture", type=str, default="lstm")
    parser.add_argument("-eo", "--epochs_offset", type=int, default=0)
    args = parser.parse_args()

    assert args.architecture in ["lstm", "transf"]

    if args.inference:
        # cannot inference without checkpoint
        assert args.load != None

    assert args.mode in ["greedy", "topp", "topk"]
    main(args)
