import torch
import json
import torch.utils.data as tud
import argparse
import os
import matplotlib.pyplot as plt

class GenerateSample:
    def __init__(self, model, i2w, sos_idx, eos_idx, device, mode, temperature, args_delimeter):
        self.model = model
        self.i2w = i2w
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.device = device
        self.mode = mode
        self.temperature = temperature
        self.args_delimeter = args_delimeter

    def __call__(self, i):
        gen = self.model.inference(
            self.sos_idx,
            self.eos_idx,
            device=self.device,
            mode=self.mode,
            temperature=self.temperature,
        )
        gen = [self.i2w[str(i)] for i in gen]
        s = "".join(gen[1:-1])
        headers = f"X:{i}\nL:1/8\nQ:120\nM:4/4\nK:C\n"
        return headers + s + f"\n{self.args_delimeter}\n"
