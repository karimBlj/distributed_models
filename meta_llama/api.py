import torch
from datetime import datetime
import torch.nn as nn
from torch.nn import functional as F
import time
from random import randint
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
# from llama import TransformerBlock, ModelArgs
from llama.model import TransformerBlock, ModelArgs
from typing import Optional
from tqdm import tqdm

app = FastAPI()


class InputData(BaseModel):
    input_tensor: list
    start_pos : int
    freqs_cis_real : list
    freqs_cis_imag : list
    mask      : Optional[list]

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def process_input(input_tensor):
    tensor = torch.tensor(input_tensor).to(device)
    # return tensor.unsqueeze(0)  # Add batch dimension
    return tensor

n_layer = 32
torch.set_default_tensor_type(torch.HalfTensor)

params = ModelArgs(
    max_seq_len=128,
    max_batch_size=4,
    # **params,
)

layers = []
for i in tqdm(range(n_layer)):
    layer = TransformerBlock(i, params)
    layers.append(layer)

@app.post("/layer_{i}/forward")
def forward_head(i : int, input_data: InputData):
    input_tensor = process_input(input_data.input_tensor)
    freq_cis_real = process_input(input_data.freqs_cis_real)
    freq_cis_imag = process_input(input_data.freqs_cis_imag)
    freq_cis = torch.complex(freq_cis_real, freq_cis_imag)
    mask = None
    if input_data.mask:
        mask = process_input(input_data.mask)
    with torch.inference_mode():
        output = layers[i].forward(input_tensor, input_data.start_pos, freq_cis, mask)
    return {"output": output.tolist()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
