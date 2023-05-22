import train


class Params:
    def __init__(self):
        self.epochs = 200
        self.hidden_size = 16
        self.embedding_size = 16
        self.layers = 2
        self.dropout = 0.2
        self.attention_heads = 1
        self.learning_rate = 0.01
        self.batch_size = 16
        self.train_ratio = 0.9
        self.create_data = False
        self.patience = 5
        self.load = None
        self.inference = False
        self.mode = "greedy"
        self.temperature = 1
        self.max_sequence_length = 256
        self.ckpt_dir = None
        self.architecture = "lstm"
        self.epochs_offset = 0
        self.create_data = False


args = Params()

# embedding
loss_dict = {}
for e in [16, 32, 64]:
    args.embedding_size = e
    args.hidden_size = e
    loss = train.main(args)
    loss_dict[e] = loss
val = sorted(loss_dict.items(), key=lambda x: x[1])[0][0]
print("Best embedding/hidden size ", val)
args.embedding_size = val
args.hidden_size = val

# layers
loss_dict = {}
for l in [2, 3, 4]:
    args.layers = l
    loss = train.main(args)
    loss_dict[l] = loss
val = sorted(loss_dict.items(), key=lambda x: x[1])[0][0]
print("Best layer num ", val)
args.layers = val

loss_dict = {}
for l in [0.01, 0.001]:
    args.learning_rate = l
    loss = train.main(args)
    loss_dict[l] = loss
val = sorted(loss_dict.items(), key=lambda x: x[1])[0][0]
print("Best learning rate ", val)
args.learning_rate = val

loss_dict = {}
for b in [16, 32, 64]:
    args.batch_size = b
    loss = train.main(args)
    loss_dict[l] = loss
val = sorted(loss_dict.items(), key=lambda x: x[1])[0][0]
print("Best batch size", val)
args.batch_size = val

print(vars(args))
with open("results.txt", "w") as f:
    f.write(str(vars(args)))
