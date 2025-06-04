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
num_stroke = 2000
num_cp = 5

canvas_height = target.shape[2]
canvas_width = target.shape[3]
stroke_renderer = BrushStrokeRenderer(num_stroke, num_cp, canvas_width, canvas_height,
                                      background_color=(0, 0, 0),
                                      feather=1.5).to(device)

x = (torch.rand(num_batch, num_stroke, 2, 1) * 0.96 + 0.02)
x[..., 0, :] *= canvas_height
x[..., 1, :] *= canvas_width
# dx1 = torch.linspace(-2, 2, num_cp).view(1, 1, 1, num_cp).repeat(num_batch, 1600, 2, 1)

dx1 = torch.randn(num_batch, 1600, 2, num_cp) * canvas_width / 1000
dx2 = torch.randn(num_batch, 400, 2, num_cp) * canvas_width / 100
dx = torch.cat([dx1, dx2], dim=1)
x = x + dx

# x = x + dx
# z = torch.rand(num_batch, num_stroke, 1, num_cp)
z = torch.linspace(0.1, 0.9, num_stroke).view(1, num_stroke, 1, 1).repeat(num_batch, 1, 1, num_cp)
w1 = torch.rand(num_batch, 1600, 1, num_cp) * 1
w2 = torch.ones(num_batch, 400, 1, num_cp) * 10
# w3 = torch.rand(num_batch, 10, 1, num_cp) * 20
w = torch.cat([w1, w2], dim=1)

c = torch.randn(num_batch, num_stroke, 4) / 50 + 2
c[..., 3] = -2
hs = torch.nn.Sigmoid()

x = x.to(device)
z = z.to(device)
w = w.to(device)
c = c.to(device)

x.requires_grad = True
w.requires_grad = True
c.requires_grad = True

perception_loss = ttools.modules.LPIPS().to(device)
cp_optimizer = Adagrad([x], lr=0.3)
cp_scheduler = MultiStepLR(cp_optimizer, [30, 60, 90], gamma=0.5)
w_optimizer = Adagrad([w], lr=0.1)
w_scheduler = MultiStepLR(w_optimizer, [30, 60, 90], gamma=0.5)
c_optimizer = Adagrad([c], lr=0.1)
color_scheduler = MultiStepLR(c_optimizer, [30, 60, 90], gamma=0.5)

for i in range(150):
    out = stroke_renderer(x, z, w.relu(), hs(c))
    p_loss = perception_loss(out, target)
    l_loss = (out - target).abs().mean()

    loss = 0.01 * p_loss + l_loss
    cp_optimizer.zero_grad()
    w_optimizer.zero_grad()
    c_optimizer.zero_grad()
    loss.backward()

    cp_optimizer.step()
    w_optimizer.step()
    c_optimizer.step()
    cp_scheduler.step()
    w_scheduler.step()
    color_scheduler.step()

    print(f'iteration {i}, loss={loss.item()}')
    if i % 10 == 0:
        plot_tensor(out[0].permute(1, 2, 0).detach().cpu(), f'iteration {i}')

print('high res output')
stroke_renderer_high_res = BrushStrokeRenderer(num_stroke, num_cp, canvas_width * 3, canvas_height * 3,
                                               background_color=(0, 0, 0),
                                               noise_intensity=1.0,
                                               feather=4.5).to(device)
out = stroke_renderer_high_res(x * 3, z, w * 3, hs(c))
skimage.io.imsave('out.jpg', out[0].permute(1, 2, 0).detach().cpu())
