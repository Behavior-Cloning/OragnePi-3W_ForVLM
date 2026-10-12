"""Independent held-out random inputs, not the four quantization examples."""
import argparse
from pathlib import Path
import numpy as np
import torch
from probe import Decoder, DIM

p=argparse.ArgumentParser()
p.add_argument('--tokens',type=int,default=32)
p.add_argument('--output',type=Path,required=True)
args=p.parse_args()
torch.manual_seed(20261010)
layers=[Decoder().eval(),Decoder().eval()]
torch.manual_seed(99173)
hidden=torch.randn(1,args.tokens,DIM)*.6
with torch.inference_mode():
    y=hidden
    for layer in layers:y=layer.full(y)
args.output.mkdir(parents=True,exist_ok=True)
hidden.numpy().astype(np.float32).tofile(args.output/'hidden.f32')
y.numpy().astype(np.float32).tofile(args.output/'reference.f32')
