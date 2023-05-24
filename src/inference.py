
import torch
import torch.nn as nn
import torch.utils.data as tud
import argparse
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

from lstm_model import *
from transf_model import *

def main(args, parameters):
    
    # set cpu as device
    device = "cpu"

    # load the vocabulary
    with open(args.vocab_file, "r") as f:
        data = json.load(f)
        w2i = data["w2i"]
        i2w = data["i2w"]
        sos_idx = w2i["<sos>"]
        eos_idx = w2i["<eos>"]

    
    parameters["vocab_size"] = len(w2i)
    
    # create a model
    # TODO kan jag inte spara hela modellen som en binär och ladda in istället?
    if parameters["architecture"] == "lstm":
        model = LSTMModel(
            vocab_size=parameters["vocab_size"],
            hidden_size=int(parameters["hs"]),
            embedding_size=int(parameters["es"]),
            num_layers=int(parameters["l"]),
            dropout=float(parameters["d"]),
            bidirectional=False
        )
    elif parameters["architecture"] == "transf":
        model = TransfModel(
            vocab_size=parameters["vocab_size"],
            hidden_size=int(parameters["hs"]),
            embedding_size=int(parameters["es"]),
            num_layers=int(parameters["l"]),
            dropout=float(parameters["d"]),
            attention_heads=int(parameters["ah"])
        )
    else:
        raise ValueError("Invalid architecture, choose lstm or transf.")
    # load the state dict
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    # inference - generating n samples
    model.eval()
    print("Inferenced samples")
    generation = ""
    for i in range(args.sample_num):
        gen = model.inference(
            sos_idx,
            eos_idx,
            device=device,
            mode=args.mode,
            temperature=args.temperature,
        )
        gen = [i2w[str(i)] for i in gen]
        s = "".join(gen[1:-1])
        headers = f"X:{i}\nL:1/8\nQ:120\nM:4/4\nK:C\n"
        generation += headers + s + f"\n{args.delimeter}\n"

        # create a folder for the generated samples
        if not os.path.exists("generated"):
            os.makedirs("generated")

        with open(f"generated/{args.sample_num}_samples_from_{args.model_name}.abc", "w") as f:
            f.write(generation)

        # TODO further preprocessing of the generated samples into a format for analyzing

def get_model_parameters_from_name(model_name):
    """
    Get model parameters from model name.
    Returns a dictionarty with the parameters.
        architecture: lstm or transf
        l: number of layers
        hs: hidden size
        es: embedding size
        d: dropout
        e: epochs
        lr: learning rate
        bs: batch size
        (transf only)
        ah: attention heads
    """
    parameters = model_name.split("_")
    parameters = [p.split("=") for p in parameters]
    # prepend 'architecture' to the first parameter
    parameters[0].insert(0, "architecture")
    parameters = {p[0]: p[1] for p in parameters}
    return parameters

if __name__ == "__main__":
    # TODO fix bug where the model is just genereating the same thing over and over again
    parser = argparse.ArgumentParser()
    # the name of the model contains all the parameters
    # e.g "lstm_l=2_es=16_hs=16_d=0.2_e=10_lr=0.01_bs=100"
    parser.add_argument("-mn", "--model_name", type=str, default=None)
    # path to checkpoint
    parser.add_argument("-ckpt", "--checkpoint", type=str, default=None)
    # path to vocab file
    parser.add_argument("-vcb", "--vocab_file", type=str, default=None)

    # generation parameters
    parser.add_argument("-m", "--mode", type=str, default="greedy")
    parser.add_argument("-t", "--temperature", type=float, default=1)
    parser.add_argument("-n", "--sample_num", type=int, default=1)

    # abc file parameters for ease of analysis
    parser.add_argument("-d", "--delimeter", type=str, default="$")
    args = parser.parse_args()

    # cannot inference without checkpoint
    assert args.model_name != None
    assert args.checkpoint != None
    assert args.vocab_file != None
    assert args.mode in ["greedy", "topp", "topk"]

    parameters = get_model_parameters_from_name(args.model_name)
    
    main(args, parameters)

"""
COPY PASTE ME SAYS THE LITTLE CODE SNIPPET BELOW :)
python src/inference.py -mn lstm_l=2_es=16_hs=16_d=0.2_e=10_lr=0.01_bs=100 \
    -ckpt lstm_l=2_es=16_hs=16_d=0.2_e=10_lr=0.01_bs=100/best.pytorch \
    -vcb data/vocab.json
    
"""
    