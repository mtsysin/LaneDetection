import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import sys


color_conversions = {
    0: (255,200,0), # Crosswalk
    1: (30,30,255), 
    2: (255,200,0), # Double white
    3: (150,250,0), # Single yellow
    4: (255,50,100), # Road curb
    5: (30,30,255),
    6: (30,255,100), # Single white
    8: (30,30,255),
    9: (250,128,114), # Direct drivable
    10: (0,206,209) # Alternate drivable
}

transform = transforms.ToPILImage()

imgs = torch.load('out/imgs.pt').clone()
seg = torch.load('out/seg.pt').clone()
pseg = torch.load('out/pseg.pt').clone()

seg[seg<=0] = 0
seg[seg>0] = 255

pseg[pseg<=0] = 0
pseg[pseg>0] = 255

image_num = 13

img13 = transform(imgs[image_num])
img13.save("out/img13.png")

# ----------------------------- GROUNDTRUTH -----------------------------

# ----------- Lane Lines -----------
lane13 = img13.copy()
for lane in range(9):
    if (lane == 7): continue # Do not add mask for background lane
    mask = transform(np.uint8(seg[image_num][lane].detach().numpy())).convert('RGBA')

    data = np.array(mask)
    red, green, blue, alpha = data.T
    white_areas = (red == 255) & (green == 255) & (blue == 255)
    data[..., :-1][white_areas.T] = color_conversions[lane]
    mask = Image.fromarray(data)

    alpha = mask.copy()
    alpha = alpha.convert('L')
    alpha.point(lambda p: 128 if p > 0 else 0)
    lane13.paste(mask, mask=alpha)

lane13.save("out/lane13_gt.png")

# ----------- Drivable Area -----------
drivable13 = img13.copy()
for drive in range(9,11):
    mask = transform(np.uint8(seg[image_num][drive].detach().numpy())).convert('RGBA')

    data = np.array(mask)
    red, green, blue, alpha = data.T
    white_areas = (red == 255) & (green == 255) & (blue == 255)
    data[..., :-1][white_areas.T] = color_conversions[drive]
    mask = Image.fromarray(data)
    mask.putalpha(128)

    alpha = mask.copy()
    alpha = alpha.convert('L')
    alpha.point(lambda p: 128 if p > 0 else 0)
    drivable13.paste(mask, mask=alpha)

drivable13.save("out/drivable13_gt.png")


# ----------------------------- PREDICTIONS -----------------------------

# ----------- Lane Lines -----------
lane13 = img13.copy()
for lane in range(9):
    if (lane == 7): continue # Do not add mask for background lane
    mask = transform(np.uint8(pseg[image_num][lane].detach().numpy())).convert('RGBA')

    data = np.array(mask)
    red, green, blue, alpha = data.T
    white_areas = (red == 255) & (green == 255) & (blue == 255)
    data[..., :-1][white_areas.T] = color_conversions[lane]
    mask = Image.fromarray(data)

    alpha = mask.copy()
    alpha = alpha.convert('L')
    alpha.point(lambda p: 128 if p > 0 else 0)
    lane13.paste(mask, mask=alpha)

lane13.save("out/lane13_pr.png")

# ----------- Drivable Area -----------
drivable13 = img13.copy()
for drive in range(9,11):
    mask = transform(np.uint8(pseg[image_num][drive].detach().numpy())).convert('RGBA')

    data = np.array(mask)
    red, green, blue, alpha = data.T
    white_areas = (red == 255) & (green == 255) & (blue == 255)
    data[..., :-1][white_areas.T] = color_conversions[drive]
    mask = Image.fromarray(data)
    mask.putalpha(128)

    alpha = mask.copy()
    alpha = alpha.convert('L')
    alpha.point(lambda p: 128 if p > 0 else 0)
    drivable13.paste(mask, mask=alpha)

drivable13.save("out/drivable13_pr.png")
