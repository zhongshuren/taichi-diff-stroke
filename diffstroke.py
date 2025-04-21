from typing import *
import torch
import torch.nn as nn

import nvdiffrast.torch as dr

import taichi as ti
from taichi.math import clamp, vec3, vec4, smoothstep, dot, tanh

vec5 = ti.types.vector(5, dtype=ti.f32)


# mat = torch.tensor([[1, 4, 1, 0],
#                     [-3, 0, 3, 0],
#                     [3, -6, 3, 0],
#                     [-1, 3, -3, 1]], dtype=torch.float32).mul(1 / 6)
mat = torch.tensor([[0, 2, 0, 0],
                    [-1, 0, 1, 0],
                    [2, -5, 4, -1],
                    [-1, 3, -3, 1]], dtype=torch.float32).mul(1 / 2)


@ti.data_oriented
class BrushStrokeRenderer(nn.Module):
    def __init__(self,
                 num_stroke,
                 num_cp,
                 canvas_width,
                 canvas_height,
                 background_color=(1.0, 1.0, 1.0),
                 rast_subdivision=16,
                 noise_intensity=0.1,
                 feather=2.0):
        super(BrushStrokeRenderer, self).__init__()
        ti.init(arch=ti.gpu, device_memory_GB=4)

        self.num_stroke = num_stroke
        self.num_cp = num_cp
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.register_buffer('mat', mat)

        res = (canvas_width, canvas_height)
        self.register_buffer('_res', torch.tensor([canvas_height, canvas_width, 1]))
        self._rast = ti.Vector.field(5, dtype=ti.i32, shape=res)
        self._coe_x = ti.Vector.field(4, dtype=ti.f32, shape=(num_stroke, num_cp - 3), needs_grad=True)
        self._coe_y = ti.Vector.field(4, dtype=ti.f32, shape=(num_stroke, num_cp - 3), needs_grad=True)
        self._coe_z = ti.Vector.field(4, dtype=ti.f32, shape=(num_stroke, num_cp - 3))
        self._coe_w = ti.Vector.field(4, dtype=ti.f32, shape=(num_stroke, num_cp - 3), needs_grad=True)
        self._color = ti.Vector.field(4, dtype=ti.f32, shape=num_stroke, needs_grad=True)
        self.background_color = vec3(*background_color)
        self.feather = ti.field(ti.f32, shape=())
        self.feather[None] = feather
        self.num_seg = ti.field(ti.f32, shape=())
        self.num_seg[None] = num_cp - 3

        self.color_buffer = ti.Vector.field(3, dtype=ti.f32, shape=res, needs_grad=True)
        self.num_frag = ti.field(int, shape=res)
        self.noise = ti.field(float, shape=res)
        self.noise.from_torch(torch.randn(res) * noise_intensity)

        self.ColorWithDepth = ti.types.struct(color=vec4, depth=ti.f32)
        self.colors_in_pixel = self.ColorWithDepth.field()
        ti.root.dense(ti.ij, res).dense(ti.k, 16).place(self.colors_in_pixel)
        ti.root.dense(ti.ij, res).dense(ti.k, 16).place(self.colors_in_pixel.grad)

        self.RootWithSeg = ti.types.struct(sid=ti.i32, segment=ti.i32, root=ti.f32)
        self.roots_in_pixel = self.RootWithSeg.field()
        ti.root.dense(ti.ij, res).dense(ti.k, 16).place(self.roots_in_pixel)

        self.gl_ctx = dr.RasterizeGLContext()
        self.rast_subdivision = rast_subdivision
        t = torch.linspace(0, 1, rast_subdivision)[1:].repeat(num_cp - 3)
        t = torch.cat([torch.tensor([0]), t])
        self.register_buffer('t', t)

        self.renderer = self.stroke_renderer_wrapper().apply

    def forward(self, control_points, depths, widths, color):
        device = control_points.device

        with torch.no_grad():
            points, tri, tri_group_size = self.get_geometry(control_points, depths, widths, device)
            rast = self.rasterize(points, tri, tri_group_size)

        v = torch.concat([control_points, depths, widths], dim=2).unfold(3, 4, 1)
        coe = torch.einsum('ef,bcpnf->bcpne', self.mat, v)  # b, c, 3, ncp - 3, 4
        coe_x, coe_y, coe_z, coe_w = torch.split(coe, 1, dim=2)

        out = self.renderer(coe_x, coe_y, coe_z, coe_w, color, rast)
        return out.permute(2, 1, 0).unsqueeze(0)

    @staticmethod
    def calc_z(inp):
        p = inp.unfold(3, 4, 1)
        return (p[..., 0] + 4 * p[..., 1] + p[..., 2]) / 6

    def interpolate_spline(self, inp, sub_step, t, device):
        b, c, m, n = inp.shape
        const = torch.ones(*t.shape).to(device)
        basis_func = torch.stack([const, t, t * t, t ** 3], dim=-1)
        basis_func = torch.matmul(basis_func, self.mat).unsqueeze(0)

        p = inp.unfold(3, 4, 1)
        p = p.unsqueeze(-2).expand(b, c, m, n - 3, sub_step - 1, 4).flatten(3, 4)
        p = torch.cat([inp[..., :4].unsqueeze(3), p], dim=3)
        interp = (basis_func * p).sum(-1)
        return interp

    def spline2points(self, cp, z, wd, device):
        v = cp[..., 1:] - cp[..., :-1]
        v = torch.cat((v[..., :1], v), dim=3)
        z = z.repeat(self.rast_subdivision - 1, 1, 1, 1, 1).permute(1, 2, 3, 4, 0).flatten(3)
        z = torch.cat([z[..., :1], z], dim=-1)
        depth = torch.cat([z[..., :1], z, z[..., -1:]], dim=-1)
        offset_left = torch.flip(v, [2]) * torch.tensor([-1, 1]).view(1, 2, 1).to(device)
        d = torch.sqrt((offset_left ** 2 + 1e-6).sum(2, keepdims=True))
        offset_left = offset_left / d * wd.clamp(1.5, 1024) * 2.0
        offset_right = offset_left * -1

        points_left = cp + offset_left
        points_begin_left = points_left[..., :1] - v[..., :1] / d[..., :1] * wd[..., :1] * 2
        points_end_left = points_left[..., -1:] + v[..., -1:] / d[..., -1:] * wd[..., -1:] * 2
        points_left = torch.cat([points_begin_left, points_left, points_end_left], dim=3)
        points_left = torch.cat([points_left, depth], dim=2)

        points_right = cp + offset_right
        points_begin_right = points_right[..., :1] - v[..., :1] / d[..., :1] * wd[..., :1] * 2
        points_end_right = points_right[..., -1:] + v[..., -1:] / d[..., -1:] * wd[..., -1:] * 2
        points_right = torch.cat([points_begin_right, points_right, points_end_right], dim=3)
        points_right = torch.cat([points_right, depth], dim=2)

        points = torch.stack((points_left, points_right), dim=4).flatten(3)
        points = points.transpose(2, 3).flatten(1, 2)
        points = (points / self._res.view(1, 1, 3)) * 2 - 1
        const = torch.ones(*points.shape[:-1], 1).to(device)

        points = torch.cat([points, const], dim=2)
        return points

    def get_geometry(self, cp, z, w, device):
        cp_interp = self.interpolate_spline(cp, self.rast_subdivision, self.t, device)
        z_interp = self.calc_z(z)
        w_interp = self.interpolate_spline(w, self.rast_subdivision, self.t, device)

        points = self.spline2points(cp_interp, z_interp, w_interp, device)

        v_id = torch.tensor(list(range(points.shape[1])), dtype=torch.int32).to(device)
        tri = v_id.view(self.num_stroke, -1).unfold(1, 3, 1).flatten(0, 1)
        tri_group_size = tri.shape[0] // self.num_stroke
        return points, tri, tri_group_size

    def rasterize(self, points, tri, tri_group_size):
        rast_layer = []
        with dr.DepthPeeler(self.gl_ctx,
                            points,
                            tri,
                            resolution=[self.canvas_width, self.canvas_height]
                            ) as peeler:
            for i in range(16):
                rast, _ = peeler.rasterize_next_layer()
                rast_layer.append(rast[..., 3])

        rast = torch.stack(rast_layer, dim=-1)
        rast = (rast - 1) // tri_group_size
        return rast

    @staticmethod
    @ti.func
    def get_dist(coe_x, coe_y, coe_w, u, v, root):
        root0 = 1.
        root1 = root
        root2 = root ** 2
        root3 = root ** 3

        _x = root0 * coe_x[0] + root1 * coe_x[1] + root2 * coe_x[2] + root3 * coe_x[3]
        _y = root0 * coe_y[0] + root1 * coe_y[1] + root2 * coe_y[2] + root3 * coe_y[3]
        _w = root0 * coe_w[0] + root1 * coe_w[1] + root2 * coe_w[2] + root3 * coe_w[3]

        _w = clamp(_w, 0, 128)

        return (_x - v) ** 2 + (_y - u) ** 2 - _w ** 2

    @ti.func
    def solve(self, coe_x, coe_y, coe_w, u, v, eps=1e-2):
        root = vec5(0.1, 0.3, 0.5, 0.7, 0.9)
        for _ in range(3):
            t_ = self.get_dist(coe_x, coe_y, coe_w, u, v, root)
            t_add_eps = self.get_dist(coe_x, coe_y, coe_w, u, v, root + eps)
            t_sub_eps = self.get_dist(coe_x, coe_y, coe_w, u, v, root - eps)
            delta = (t_add_eps - t_sub_eps) / (t_add_eps + t_sub_eps - t_ * 2) * eps / 2
            root = clamp(root - delta, 0., 1.)
        return root

    @ti.kernel
    def get_root(self):
        for u, v in self.color_buffer:
            self.num_frag[u, v] = 0
            last_sid = -1
            for i in range(16):
                sid = self._rast[u, v][i]
                if sid == -1:
                    break
                if sid == last_sid:
                    continue
                last_sid = sid
                min_dist = 4096.
                rws = self.RootWithSeg(sid, 0, 0.)

                for seg in range(4):
                    coe_x = self._coe_x[sid, seg]
                    coe_y = self._coe_y[sid, seg]
                    coe_w = self._coe_w[sid, seg]
                    root = self.solve(coe_x, coe_y, coe_w, u, v)
                    tmp = self.get_dist(coe_x, coe_y, coe_w, u, v, root)
                    for j in range(5):
                        if tmp[j] < min_dist:
                            min_dist = tmp[j]
                            rws.root = root[j]
                            rws.segment = seg

                self.roots_in_pixel[u, v, self.num_frag[u, v]] = rws
                self.num_frag[u, v] += 1

    @ti.kernel
    def get_frag(self):
        for u, v in self.color_buffer:
            l = self.num_frag[u, v]
            for i in range(l):
                sid = self.roots_in_pixel[u, v, i].sid
                seg = self.roots_in_pixel[u, v, i].segment
                root = self.roots_in_pixel[u, v, i].root

                coe_x = self._coe_x[sid, seg]
                coe_y = self._coe_y[sid, seg]
                coe_z = self._coe_z[sid, seg]
                coe_w = self._coe_w[sid, seg]

                rs = vec4(1., root, root ** 2, root ** 3)

                _x = dot(coe_x, rs)
                _y = dot(coe_y, rs)
                _z = dot(coe_z, rs)
                _w = dot(coe_w, rs)

                _w = clamp(_w, 0, 128)

                dist = ti.sqrt((_x - v) ** 2 + (_y - u) ** 2) - _w + self.noise[u, v]
                # flow = tanh(40 * (root + seg)) * tanh(40 * (self.num_seg[None] - root - seg))
                alpha = smoothstep(-self.feather[None], _w / 2, -dist) * self._color[sid].a
                frag = vec4(self._color[sid].rgb, alpha)
                depth = _z + (_w - dist)
                self.colors_in_pixel[u, v, i] = self.ColorWithDepth(color=frag, depth=depth)

    @ti.kernel
    def bubble_sort(self):
        for u, v in self.color_buffer:
            l = self.num_frag[u, v]
            for i in range(l - 1):
                for j in range(l - 1 - i):
                    if self.colors_in_pixel[u, v, j].depth > self.colors_in_pixel[u, v, j + 1].depth:
                        tmp = self.colors_in_pixel[u, v, j]
                        self.colors_in_pixel[u, v, j] = self.colors_in_pixel[u, v, j + 1]
                        self.colors_in_pixel[u, v, j + 1] = tmp

    @ti.kernel
    def get_color(self):
        for u, v in self.color_buffer:
            color = self.background_color
            l = self.num_frag[u, v]
            for i in range(l):
                next_color = self.colors_in_pixel[u, v, l - 1 - i].color
                color = (1 - next_color.a) * color + next_color.a * next_color.rgb
            self.color_buffer[u, v] = color

    @ti.kernel
    def init(self):
        for u, v in self.color_buffer:
            self.num_frag[u, v] = 0

    def stroke_renderer_wrapper(self):
        outer_self = self

        class stroke_renderer(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any,
                        coe_x=None,
                        coe_y=None,
                        coe_z=None,
                        coe_w=None,
                        color=None,
                        rast=None,
                        *args: Any, **kwargs: Any) -> Any:

                outer_self._coe_x.from_torch(coe_x[0].squeeze(1))
                outer_self._coe_y.from_torch(coe_y[0].squeeze(1))
                outer_self._coe_z.from_torch(coe_z[0].squeeze(1))
                outer_self._coe_w.from_torch(coe_w[0].squeeze(1))
                outer_self._color.from_torch(color[0])
                outer_self._rast.from_torch(rast[0])

                outer_self.init()
                outer_self.get_root()
                outer_self.get_frag()
                outer_self.bubble_sort()
                outer_self.get_color()

                ctx.coe_x = coe_x
                ctx.coe_y = coe_y
                ctx.coe_w = coe_w

                return outer_self.color_buffer.to_torch(device=coe_x.device)

            @staticmethod
            def backward(ctx: Any, grad_output=None, *args: Any, **kwargs: Any) -> Any:
                outer_self.color_buffer.grad.from_torch(grad_output)

                outer_self.get_color.grad()
                outer_self.bubble_sort.grad()
                outer_self.get_frag.grad()

                coe_x_grad = outer_self._coe_x.grad.to_torch(device=grad_output.device).unsqueeze(0).unsqueeze(2)
                coe_y_grad = outer_self._coe_y.grad.to_torch(device=grad_output.device).unsqueeze(0).unsqueeze(2)
                coe_w_grad = outer_self._coe_w.grad.to_torch(device=grad_output.device).unsqueeze(0).unsqueeze(2)
                color_grad = outer_self._color.grad.to_torch(device=grad_output.device).unsqueeze(0)

                return coe_x_grad, coe_y_grad, None, coe_w_grad, color_grad, None

        return stroke_renderer
