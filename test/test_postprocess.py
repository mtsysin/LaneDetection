import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from model.YoloMulti import YoloMulti
import unittest
from postprocess import *
from bdd100k import *
from train import ROOT, USE_PARALLEL, INPUT_IMG_TRANSFORM

device = torch.device('cuda')
print(torch.cuda.is_available()) 
print(f"CUDA device: {torch.cuda.current_device()}")
print(f"CUDA device count: {torch.cuda.device_count()}")

class TestPostprocess(unittest.TestCase):
    def test_postprocess_random(self):
        """Try to run this for random prediction"""
        pred1 = torch.rand(16, 3, 10, 20, 18)
        pred2 = torch.rand(16, 3, 10, 20, 18)
        pred3 = torch.rand(16, 3, 10, 20, 18)
        pred = [pred1, pred2, pred3]

        nms_b = get_bboxes(pred, 0.7, 0.7)

    def test_postprocess_on_dataset_output(self):
        """Try to run this for sample prediction"""
        # Create dataset
        dataset = BDD100k(
            root = BDD_100K_ROOT,
            transform = transforms.Compose([
                transforms.Resize((384, 640), interpolation=transforms.InterpolationMode.NEAREST)
            ])
        )
        loader = DataLoader(dataset, batch_size=4)

        C = len(CLASS_DICT)
        iterator = iter(loader)
        for _ in range(1):
            image, label, _ = next(iterator)

        for l in label:
            assert not torch.isnan(l).any()
        assert not torch.isnan(image).any()


        print("label[0].size()", label[0].size())
        nms_b = get_bboxes(label, 0.7, 0.7, true_prediction=False) # Get bboxes for a the first image in a btach

        print(len(nms_b))
        print(nms_b[0].size())
        print(nms_b[0])

        print(image[0, 2:20, 3:30])

        image = draw_bbox(image, dets=nms_b)
        image = (image/255.)
        torchvision.utils.save_image(image[0], "out/test_postprocess_on_dataset_output.png")

    def test_postprocess_on_simple_pretrained_model(self):

        index = 0

        """Try to run this for sample prediction"""
        # Create dataset
        dataset = BDD100k(
            root = BDD_100K_ROOT,
            transform = INPUT_IMG_TRANSFORM
        )
        loader = DataLoader(dataset, batch_size=1)

        C = len(CLASS_DICT)
        image, _, _ = next(iter(loader))

        model = YoloMulti().to(device)
        if USE_PARALLEL:
            model= torch.nn.DataParallel(model)
            model.to(device)
        model.load_state_dict(torch.load(ROOT+'/model.pt'))
        model.eval()

        image = image.to(device)

        label, _ = model(image)
        label = [l.cpu() for l in label]
        image = image.cpu()

        print(f"label[0].size()", label[0].size())
        nms_b = get_bboxes(label, 0.2, 0.9, true_prediction=True) # Get bboxes for a the first image in a btach

        print(f"len(nms_b)", len(nms_b))
        print(f"nms_b[{index}].size()", nms_b[index].size())
        print(f"nms_b[{index}]", nms_b[index])

        print(image[0, 2:20, 3:30])
        image = (image + 1.) / 2. * 255.
        image = draw_bbox(image, dets=nms_b)
        image = image / 255.
        torchvision.utils.save_image(image[index], "out/test_postprocess_on_simple_pretrained_model.png")

