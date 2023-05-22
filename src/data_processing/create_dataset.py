from tqdm import tqdm
import os
from collections import defaultdict

ignore = [
    "^F,,",
    "e'",
    "^A,,",
    "=A",
    "=A,",
    "^G,,",
    "^C,",
    "^g",
    "=D",
    "=d",
    "^a",
    "g'",
    "B,,,,",
    "f'",
    "^d'",
    "c''",
    "a'",
    "^c'",
    "^D,,",
    "=A,,",
    "=G,",
    "=C",
    "=F",
    "g''",
    "b'",
    "=D,",
    "=a",
    "e''",
    "=F,",
    "^f'",
    "=G",
]

max_length = 256

data = ""
for path_to_file in tqdm(os.listdir("abc")):
    file_tokens = defaultdict(int)
    skip = False

    # open the file
    with open(f"abc/{path_to_file}", "r") as file:
        tokens = file.read().split(" ")

        if len(tokens) > max_length:
            continue

        # check for tokens to ignore
        for tok in ignore:
            if tok in tokens:
                skip = True
                break

        # skip the file
        if skip:
            continue

        # split file_tokens by space and remove empty strings
        tokens = list(filter(None, tokens))

        data += " ".join(tokens) + "\n"

with open("data/dataset.txt", "w") as f:
    f.write(data)
