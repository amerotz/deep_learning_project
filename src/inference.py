import torch
import torch.nn as nn
import torch.utils.data as tud
import argparse
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import mlflow
import tempfile

from lstm_model import *
from transf_model import *

def main(args, model):
    
    # set cpu as device
    device = "cpu"

    # load the vocabulary
    with open(args.vocab_file, "r") as f:
        data = json.load(f)
        w2i = data["w2i"]
        i2w = data["i2w"]
        sos_idx = w2i["<sos>"]
        eos_idx = w2i["<eos>"]

    #vocab_size = len(w2i)

    # inference - generating n samples
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

        with tempfile.NamedTemporaryFile(prefix="samples_", suffix=".txt") as temp:
            # TODO this generates a lot of empty files, why??
            
            # Write the generation string to the file
            temp.write(generation.encode())
            # Remember the file name
            temp_name = temp.name
            # Log the file as an artifact of the MLflow run
            mlflow.log_artifact(temp_name, "generated_samples")

        # TODO further preprocessing of the generated samples into a format for analyzing
        # convert abc to midi and log the number of errors
        # convert the abc to correct format (as the dataset)

        
    

if __name__ == "__main__":
    # TODO fix bug where the model is just genereating the same thing over and over again
    parser = argparse.ArgumentParser()
    # MLFlow RunID of the traning - the runid is used to load the model
    parser.add_argument("-en", "--experiment_name", type=str, default=None)
    parser.add_argument("-r", "--run_id", type=str, default=None)

    # path to vocab file
    parser.add_argument("-vcb", "--vocab_file", type=str, default=None)

    # generation parameters
    parser.add_argument("-m", "--mode", type=str, default="greedy")
    parser.add_argument("-t", "--temperature", type=float, default=1)
    parser.add_argument("-n", "--sample_num", type=int, default=10)

    # abc file parameters for ease of analysis
    parser.add_argument("-d", "--delimeter", type=str, default="$")
    args = parser.parse_args()

    # cannot inference without checkpoint
    assert args.run_id != None
    assert args.experiment_name != None
    assert args.vocab_file != None
    assert args.mode in ["greedy", "topp", "topk"]

    # load the model from the runid
    model_uri = "runs:/" + args.run_id + "/models"
    # TODO ska inte behöva flytta den till cpu här egentligen
    loaded_model = mlflow.pytorch.load_model(model_uri).to("cpu")

    # set the correct experiment
    mlflow.set_experiment(args.experiment_name)
    
    # set the correct run
    mlflow.start_run(run_id=args.run_id)
    
    main(args, loaded_model)

    # end experiment
    mlflow.end_run()

"""
COPY PASTE ME SAYS THE LITTLE CODE SNIPPET BELOW :)
python src/inference.py --experiment_name "My Experiment" \
                        --run_id "e0a0b588836e469184c262603bb19617" \
                        --vocab_file "data/vocab.json" \
                        --mode "topp" \
                        --temperature 0.5
    
"""

"""
run id glamouras goat
e87974ba799c4864b8da325bdbebcd2a
"""
    