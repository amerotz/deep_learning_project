# Project for DD2424 Deep Learning in Data Science
Authors: Linus Östlund and Marco Amoretti

## Project description

* Something on the dataset (cite their work)
* Something on the architecture
* Something on Molly

## How to use

### Create Virtual Environment

Using either `conda` or `pip`, create a virtual environment. Activate it and then call:
```
pip install -r requirements.txt
```

### Generate data

To generate training data from `dataset.pkl` use:
```
python3 src/data_processing/preprocessing.py
python3 src/data_processing/create_data.py
```

This will convert the pickled dataset to midi files, the midi files to abc notation and then perform a cleaning step that will:
- remove files with low-occurence tokens (less than 100 occurrences in the whole dataset);
- reformat all training data in one file (one line per piece, with space-separated tokens).

### Train the model (LSTM)

To train a model call:
```
python3 src/train.py
```

The command line arguments provide costumization for model size and layers, number of epochs, patience for early stopping and so on.
A checkpoint for each epoch will be saved in `./ckpts/`.

The flag `-ld {CKPT}` loads a previous checkpoint and continues training from there. The size of the loaded model has to match the provided command line arguments (e.g. if the checkpoint has 3 layers add `-l 3` and similar).

### Inference the model

To generate from a checkpoint use:
```
python3 src/train.py -i -ld {CKPT} -m {'greedy', 'topk', 'topp'} -t {TEMPERATURE}
```
The model can be sampled with greedy, Top-P and Top-K sampling. Generation starts from an empty sequence starting with '<sos>' and continues until either the maximum sequence length is reached or a '<eos>' token is generated.

Output is printed to the console. To listen to the output:
- create an ABC-file with the following fields:
```
X:0
M:4/4
L:1/8
Q:120
K:C
```
- copy all tokens between '<sos>' and '<eos>' and paste them after `K:C`;
- invoke `abc2midi <abc_file> -o <midi_file>`;
- listen in your program of choice or with `timidity <midi_file>`

