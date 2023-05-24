from tqdm import tqdm
from lstm_model import *
from transf_model import *
from dataset import *

# load vocab
dataset = MusicDataset(
    data_file="data/dataset.json",
    vocab_file="data/vocab.json",
    max_sequence_length=256,
    create_data=False,
)
vocab_size = dataset.vocab_size

# load model
model = LSTMModel(
    vocab_size=vocab_size,
    hidden_size=64,
    embedding_size=64,
    num_layers=2,
    dropout=0.2,
    bidirectional=False,
)
ckpt = "lstm_l=2_es=64_hs=64_d=0.2_e=200_lr=0.001_bs=16/best.pytorch"
"""
# load model
model = TransfModel(
    vocab_size=vocab_size,
    hidden_size=64,
    embedding_size=64,
    num_layers=3,
    dropout=0.2,
    attention_heads=4,
)
ckpt = "transf_l=3_es=64_hs=64_d=0.2_e=200_lr=0.001_bs=16_ah=4/best.pytorch"
"""
model.load_state_dict(torch.load(ckpt))


device = "cpu"
if torch.cuda.is_available():
    print("Using CUDA.")
    model = model.cuda()
    device = "cuda"

# sample model
model.eval()
generation = []

for i in range(516):
    gen = model.inference(
        dataset.sos_idx,
        dataset.eos_idx,
        device=device,
        mode="topp",
        temperature=1,
    )
    gen = [dataset.i2w[str(i)] for i in gen[1:-1]]
    gen = gen + ["\n"]
    generation.extend(gen)
print(generation[:100])

# read training data
with open("data/training_data.txt", "r") as f:
    training_data = f.read()

bars = training_data.split("|")
bars = [len(list(filter(None, b.split(" ")))) for b in bars]
avg_bar_len = sum(bars) / len(bars)
print(f"Average bar length: {avg_bar_len}")

training_data = training_data.split(" ")
print(training_data[:100])
with open("data/validation_data.txt", "r") as f:
    validation_data = f.read().split(" ")


def ngram_similarity(tokens_o, tokens_p, n):
    tokens_p = list(filter(None, tokens_p))
    tokens_o = list(filter(None, tokens_o))
    n -= 1
    trigrams_o = defaultdict(bool)
    for i in tqdm(range(len(tokens_o) - n)):
        t = tuple(tokens_o[i : i + n])
        trigrams_o[t] = True

    s = 0
    trigrams_p = defaultdict(bool)
    for i in tqdm(range(len(tokens_p) - n)):
        t = tuple(tokens_p[i : i + n])
        if trigrams_o[t]:
            if not trigrams_p[t]:
                s += 1
        trigrams_p[t] = True

    # containment measure
    C = s / len(trigrams_p.keys())
    return C


vals = [avg_bar_len / 2, avg_bar_len, avg_bar_len * 1.5, avg_bar_len * 2]
for n in vals:
    n = int(n)
    v = ngram_similarity(training_data, generation, n)
    print(f"Containment for n={n}: {v}")
