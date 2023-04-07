import torch
from model.YoloMulti import YoloMulti
import unittest
from torchinfo import summary

class TestModel(unittest.TestCase):
    def test_model_ouptut(self):
        torch.cuda.empty_cache()
        device = torch.device('cuda:1')
        BATCH = 8
        # print(torch.cuda.memory_summary(device=None, abbreviated=False))
        model = YoloMulti().to(device)
        print(next(model.parameters()).device)
        # print(torch.cuda.memory_summary(device=None, abbreviated=False))

        summary(model, input_size=(BATCH, 3, 384, 640), depth=6)

        x = torch.randn(BATCH, 3, 384, 640)
        x = x.to(device)

        det, seg = model(x)
        print(f'Detection scale 1 {det[0].shape}')
        print(f'Detection scale 2 {det[1].shape}')
        print(f'Detection scale 3 {det[2].shape}')
        print(f'Lane {seg.shape}')