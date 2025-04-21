import sys

import torch

from torch.optim import Adagrad, Adam
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
import torchvision.transforms as tr

from diffstroke import BrushStrokeRenderer
import skimage.io
import skimage.transform
import numpy as np
from utils import plot_tensor, calc_grad

import ttools.modules

torch.manual_seed(1001)
device = torch.device('cuda')

image = skimage.io.imread('apple.jpg')
scale = 400 / max(*image.shape[:2])
size = (int(image.shape[0] * scale), int(image.shape[1] * scale))
image = skimage.transform.resize(image, size, anti_aliasing=True)
image = np.array(image).astype(np.float32)
target = torch.from_numpy(image).to(device)
target = target.unsqueeze(0)
target = target.permute(0, 3, 1, 2)
target = target[:, :3, ...]

image2 = skimage.io.imread('girl-on-a-divan.jpg')
image2 = skimage.transform.resize(image2, size, anti_aliasing=True)
image2 = np.array(image2).astype(np.float32)
style = torch.from_numpy(image2).to(device)
style = style.unsqueeze(0)
style = style.permute(0, 3, 1, 2)
style = style[:, :3, ...]

num_batch = 1
num_stroke = 1000
num_cp = 4

canvas_height = target.shape[2]
canvas_width = target.shape[3]
stroke_renderer = BrushStrokeRenderer(num_stroke, num_cp, canvas_width, canvas_height,
                                      background_color=(1, 1, 1),
                                      feather=1.5).to(device)

x = (torch.rand(num_batch, num_stroke, 2, 1) * 0.96 + 0.02)
x[..., 0, :] *= canvas_height
x[..., 1, :] *= canvas_width
# dx1 = torch.linspace(-2, 2, num_cp).view(1, 1, 1, num_cp).repeat(num_batch, 6000, 2, 1)

dx1 = torch.randn(num_batch, 1000, 2, num_cp) * canvas_width / 100
dx2 = torch.randn(num_batch, 0, 2, num_cp) * canvas_width / 100

dx = torch.cat([dx1, dx2], dim=1)
x = x + dx

z = torch.linspace(0.1, 0.9, num_stroke).view(1, num_stroke, 1, 1).repeat(num_batch, 1, 1, num_cp)

w1 = torch.ones(num_batch, 1000, 1, num_cp) * 1
w2 = torch.ones(num_batch, 0, 1, num_cp) * 10
w = torch.cat([w1, w2], dim=1)

c = torch.randn(num_batch, num_stroke, 4) / 10 - 3
c[..., 3] = 1
hs = torch.nn.Hardsigmoid()

x = x.to(device)
z = z.to(device)
w = w.to(device)
c = c.to(device)

x.requires_grad = True

perception_loss = ttools.modules.LPIPS().to(device)
cp_optimizer = Adagrad([x], lr=0.1)
cp_scheduler = MultiStepLR(cp_optimizer, [30, 60], gamma=0.5)

for i in range(150):
    out = stroke_renderer(x, z, w.relu(), hs(c))
    p_loss = perception_loss(out, target)
    l_loss = (out - target).pow(2).mean()

    loss = 0.01 * p_loss + l_loss
    cp_optimizer.zero_grad()
    loss.backward()

    cp_optimizer.step()
    cp_scheduler.step()

    print(f'iteration {i}, loss={loss.item()}')
    if i % 10 == 0:
        plot_tensor(out[0].permute(1, 2, 0).detach().cpu(), f'iteration {i}')

print('high res output')
stroke_renderer_high_res = BrushStrokeRenderer(num_stroke, num_cp, canvas_width * 3, canvas_height * 3,
                                               background_color=(1, 1, 1),
                                               noise_intensity=1.0,
                                               feather=4.5).to(device)
out = stroke_renderer_high_res(x * 3, z, w * 3, hs(c))
skimage.io.imsave('out.jpg', out[0].permute(1, 2, 0).detach().cpu())
