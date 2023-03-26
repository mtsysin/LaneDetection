import torch
from model import YoloMulti

x = torch.randn(2, 3, 416, 416)
model = YoloMulti()
det, seg = model(x)
print(f'Detection scale 1 {det[0].shape}')
print(f'Detection scale 2 {det[1].shape}')
print(f'Detection scale 3 {det[2].shape}')
print(f'Lane {seg.shape}')