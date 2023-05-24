import torch
import torch.utils.data as tud
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict


class MusicDataset(tud.Dataset):
    def __init__(
        self,
        data_file: str,
        max_sequence_length: int,
        vocab_file=None,
        create_data=True,
    ):
        """
        Create the dataset. If `create_data==True`, `data_file`
        will be used to create the dataset, else the data and vocabs
        will be loaded from `data_file` and `vocab_file`.
        """
        super().__init__()

        self.max_sequence_length = max_sequence_length
        if create_data:
            self.w2i = defaultdict(int)
            self.i2w = defaultdict(str)
            self.data = []
            self._create_data(data_file)
        else:
            self._load_data(data_file, vocab_file)

    def _create_data(self, data_file: str):
        """
        Create dataset from `data_file`. In particular:
        - create w2i and i2w
        - create actual dataset input and target pairs
        """

        print("Creating dataset...")
        with open(data_file, "r") as f:
            # read the file and split in lines
            # 1 line = 1 piece
            lines = f.read().split("\n")

            # filter empty strings
            lines = list(filter(None, lines))

            # index pad, sos and eos tokens
            tok_id = 0
            self.w2i["<pad>"] = tok_id
            self.i2w[tok_id] = "<pad>"

            tok_id += 1
            self.w2i["<sos>"] = tok_id
            self.i2w[tok_id] = "<sos>"

            tok_id += 1
            self.w2i["<eos>"] = tok_id
            self.i2w[tok_id] = "<eos>"

            tok_id += 1

            print("Creating w2i and i2w...")
            # create w2i and i2w
            for piece in tqdm(lines):
                tokens = piece.split(" ")
                tokens = list(filter(None, tokens))

                for tok in tokens:
                    # add token to dicts
                    if not tok in self.w2i:
                        self.w2i[tok] = tok_id
                        self.i2w[tok_id] = tok
                        tok_id += 1

            # add the pieces to data using w2i
            # add <sos> at the start
            # add <eos> at the end
            # pad with <pad> until max length
            print("Indexing pieces...")
            for piece in tqdm(lines):
                tokens = piece.split(" ")
                tokens = list(filter(None, tokens))

                # input
                input = ["<sos>"] + tokens
                input = input[: self.max_sequence_length]

                # target
                target = tokens[: self.max_sequence_length - 1]
                target = target + ["<eos>"]

                # pad everything
                length = len(input)
                input.extend(["<pad>"] * (self.max_sequence_length - length))
                target.extend(["<pad>"] * (self.max_sequence_length - length))

                # convert to indexes
                input = [self.w2i.get(w) for w in input]
                target = [self.w2i.get(w) for w in target]

                # add dict to data
                item = {
                    "input": input,
                    "target": target,
                }
                self.data.append(item)

        # save data to json
        print("Writing data to json...")
        with open("data/dataset.json", "w") as f:
            data = json.dumps(self.data)
            f.write(data)

        # save vocabs to json
        print("Writing vocabs to json...")
        with open("data/vocab.json", "w") as f:
            vocab = {"w2i": self.w2i, "i2w": self.i2w}
            data = json.dumps(vocab)
            f.write(data)

        print("Done")

    @property
    def sos_idx(self):
        return self.w2i["<sos>"]

    @property
    def eos_idx(self):
        return self.w2i["<eos>"]

    @property
    def pad_idx(self):
        return self.w2i["<pad>"]

    @property
    def vocab_size(self):
        return len(self.w2i)

    def _load_data(self, data_file: str, vocab_file: str):
        """
        Load data and vocab from json files.
        """
        with open(data_file, "r") as f:
            self.data = json.load(f)

        with open(vocab_file, "r") as f:
            data = json.load(f)
            self.w2i = data["w2i"]
            self.i2w = data["i2w"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            torch.LongTensor(self.data[idx]["input"]),
            torch.LongTensor(self.data[idx]["target"]),
        )


torch.manual_seed(0)
"""
dataset = MusicDataset("data/dataset.txt", 128)
"""
dataset = MusicDataset(
    data_file="data/dataset.json",
    vocab_file="data/vocab.json",
    max_sequence_length=128,
    create_data=False,
)
train_data, val_data = tud.random_split(dataset, [0.9, 1 - 0.9])

data = ""
for seq in train_data:
    x = seq[0]
    x = x[x != 0].detach().numpy()
    s = " ".join([dataset.i2w[str(i)] for i in x[1:]])
    data += s + " \n "

with open("data/training_data.txt", "w") as f:
    f.write(data)

data = ""
for seq in val_data:
    x = seq[0]
    x = x[x != 0].detach().numpy()
    s = " ".join([dataset.i2w[str(i)] for i in x[1:]])
    data += s + " \n "

with open("data/validation_data.txt", "w") as f:
    f.write(data)
