import torch
import time
import torch.nn as nn
import torch.utils.data as tud
import argparse
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from lstm_model import *
from transf_model import *
from dataset import *


def main(args):
    print("Loading data...")

    torch.manual_seed(0)

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

    print("Creating model...")

    # model, optimizer, loss
    model = LSTMModel(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        embedding_size=args.embedding_size,
        num_layers=args.layers,
        dropout=args.dropout,
        bidirectional=False,
    )
    model.eval()
    print(model)

    print("Creating splits...")
    train_data, val_data = tud.random_split(
        dataset, [args.train_ratio, 1 - args.train_ratio]
    )

    # load previous checkpoint
    if args.load != None:
        print(f"Loading checkpoint {args.load}")
        model.load_state_dict(torch.load(args.load))

    # check gpu
    device = "cpu"

    model_name = f"{args.architecture}_l={args.layers}_es={args.embedding_size}_hs={args.hidden_size}_d={args.dropout}_e={args.epochs}_lr={args.learning_rate}_bs={args.batch_size}"

    """
    # self similarity
    # mat = model.embedding.weight
    mat = model.prediction_layer.weight
    mat = mat.detach().numpy()
    p = 2
    out = np.dot(mat, mat.T)
    norm = np.outer(
        np.linalg.norm(mat, axis=1, ord=p), np.linalg.norm(mat.T, axis=0, ord=p)
    )
    print(norm.shape)
    out = np.multiply(
        out,
        1 / norm,
    )
    out /= np.max(out)
    print(out.shape)

    plt.imshow(out)
    labels = [dataset.i2w[str(i)] for i in range(out.shape[0])]

    plt.xlabel(labels)
    plt.ylabel(labels)
    plt.show()

    plt.clf()
    """

    torch.manual_seed(time.time())
    gen = model.inference(
        dataset.sos_idx,
        dataset.eos_idx,
        device=device,
        mode=args.mode,
        temperature=args.temperature,
    )
    sample_x = torch.IntTensor(gen)
    """
    sample_x, sample_y = val_data[69]
    """
    print(sample_x)
    logits = model(sample_x.unsqueeze(0))
    probs = model.softmax(logits).squeeze(0)
    probs = probs.detach().numpy().T

    probs /= np.max(probs)

    plt.imshow(probs)
    plt.show()

    """
    # inference at end of training or because args.inference
    if args.inference:
        model.eval()
        print("Inferenced samples")
        generation = ""
        for i in range(args.sample_num):
            gen = model.inference(
                dataset.sos_idx,
                dataset.eos_idx,
                device=device,
                mode=args.mode,
                temperature=args.temperature,
            )
            gen = [dataset.i2w[str(i)] for i in gen]
            s = "".join(gen[1:-1])
            headers = f"X:{i}\nL:1/8\nQ:120\nM:4/4\nK:C\n"
            generation += headers + s + "\n"
            with open(f"generated_{model_name}.abc", "w") as f:
                f.write(generation)

    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int, default=1)
    parser.add_argument("-hs", "--hidden_size", type=int, default=16)
    parser.add_argument("-es", "--embedding_size", type=int, default=16)
    parser.add_argument("-l", "--layers", type=int, default=2)
    parser.add_argument("-dp", "--dropout", type=float, default=0.2)
    parser.add_argument("-ah", "--attention_heads", type=int, default=2)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.01)
    parser.add_argument("-bs", "--batch_size", type=int, default=100)
    parser.add_argument("-tr", "--train_ratio", type=float, default=0.9)
    parser.add_argument("-cd", "--create_data", action="store_true")
    parser.add_argument("-p", "--patience", type=int, default=5)
    parser.add_argument("-ld", "--load", type=str, default=None)
    parser.add_argument("-i", "--inference", action="store_true")
    parser.add_argument("-m", "--mode", type=str, default="greedy")
    parser.add_argument("-t", "--temperature", type=float, default=1)
    parser.add_argument("-ml", "--max_sequence_length", type=int, default=512)
    parser.add_argument("-ckd", "--ckpt_dir", type=str, default=None)
    parser.add_argument("-arch", "--architecture", type=str, default="lstm")
    parser.add_argument("-eo", "--epochs_offset", type=int, default=0)
    parser.add_argument("-n", "--sample_num", type=int, default=1)
    args = parser.parse_args()

    assert args.architecture in ["lstm", "transf"]

    if args.inference:
        # cannot inference without checkpoint
        assert args.load != None

    assert args.mode in ["greedy", "topp", "topk"]
    main(args)
