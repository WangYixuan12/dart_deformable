---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.6
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
import cv2
import torch
from torch import optim
from torch.autograd.functional import jacobian
from PIL import Image
import OpenEXR, Imath
import numpy as np
from matplotlib import pyplot as plt
import time
```

```python
def construct_xy(H, W, cuda=False):
    xy = torch.zeros(2, H, W)
    if cuda:
        xy = xy.cuda()
    xy[0] = torch.arange(W).reshape(1, W).repeat((H, 1)) # x
    xy[1] = torch.arange(H-1, -1, -1).reshape(H, 1).repeat((1, W)) # y
    return xy
```

```python
def dist_to_line_2d(xy, c1, c2):
    '''
    Input:
        xy: (2, H, W) torch tensor - xy(h, w) is the coordinate
        c1: (2,) torch tensor - (x,y) of one end
        c2: (2,) torch tensor - (x,y) of another end
    '''
    _, H, W = xy.shape
    xy_c1 = xy - c1.reshape(2,1,1)
    c12 = (c2 - c1).reshape(2,1,1)
    c12_length = c12.norm()
    c12 = c12/c12_length
    closest_dist = torch.clamp(torch.clamp(torch.sum(xy_c1 * c12, dim=0), min=0), max=c12_length)
    closest_pt = c1.reshape(2,1,1) + closest_dist * c12
    dist = torch.norm(xy - closest_pt, dim = 0)
    return dist
```

```python
def q_to_c_2d(q, H, W, cuda=False):
    l = 20
    n = q.shape[0]-1
    c = torch.zeros((n,2))
    if cuda:
        c = c.cuda()
    c[0, 0] = q[0]
    c[0, 1] = q[1]
    for i in range(1, n):
        c[i, 0] = c[i-1, 0] + l*torch.cos(q[i+1])
        c[i, 1] = c[i-1, 1] + l*torch.sin(q[i+1])
#     print(c)
    return c
```

```python
def render(q, H, W):
    sigma = 3
    c = q_to_c_2d(q, H, W)
    xy = construct_xy(H, W)
    prev_dist = dist_to_line_2d(xy.clone(), c[0], c[1])
    for i in range(c.shape[0]-1):
        dist = torch.minimum(dist_to_line_2d(xy.clone(), c[i], c[i+1]), prev_dist)
        prev_dist = dist
#     dist = torch.exp(-dist * dist/(sigma**2))
    dist = dist * dist
    return dist
```

```python
def loss_fn(q, H, W, mask):
    dist = render(q, H, W)
#     plt.imshow(dist[mask].detach().numpy())
#     print(dist[mask].detach().numpy())
#     plt.show()
    return torch.sum(dist[mask])#+torch.sum(-dist[~mask])
```

```python
def exr_to_np(path):
    y_resolution = 540
    x_resolution = 810
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    
    rgb_exr = OpenEXR.InputFile(path)
    rgb_dw = rgb_exr.header()['dataWindow']
    rgb_size = (rgb_dw.max.x - rgb_dw.min.x + 1, rgb_dw.max.y - rgb_dw.min.y + 1)
    r_str = rgb_exr.channel('R', pt)
    g_str = rgb_exr.channel('G', pt)
    b_str = rgb_exr.channel('B', pt)
    rgb_img = np.zeros((y_resolution, x_resolution, 3))
    r_ch = Image.frombytes("F", rgb_size, r_str)
    g_ch = Image.frombytes("F", rgb_size, g_str)
    b_ch = Image.frombytes("F", rgb_size, b_str)
    rgb_img[: ,:, 2] = np.array(r_ch.getdata()).reshape(y_resolution, x_resolution)
    rgb_img[: ,:, 1] = np.array(g_ch.getdata()).reshape(y_resolution, x_resolution)
    rgb_img[: ,:, 0] = np.array(b_ch.getdata()).reshape(y_resolution, x_resolution)
    min_val = rgb_img.min()
    max_val = rgb_img.max() 
    rgb_img = ((rgb_img - min_val)/(max_val - min_val) * 255.0).astype(np.uint8)
    return rgb_img
```

```python
class tracker:
    def __init__(self, H, W, cuda=False):
        self.H = H
        self.W = W
        self.cuda = cuda
        
    def set_obs(self, mask, rgb_np=None, subsample=False):
        if subsample:
#             N = mask.nonzero().shape[0]
#             n = 100
#             rand_idx = np.random.choice(N, n)
#             sub = torch.zeros_like(mask)
#             sub[mask.nonzero()[rand_idx][:, 0], mask.nonzero()[rand_idx][:, 1]] = True
#             self.mask = sub
#             if self.cuda:
#                 self.mask = sub.cuda()
            import point_cloud_utils as pcu

            # v is a nv by 3 NumPy array of vertices
            # n is a nv by 3 NumPy array of vertex normals
            # n is a nv by 4 NumPy array of vertex colors
            N = mask.nonzero().shape[0]
            print("Before subsample:", N)
            v = np.zeros((N, 3))
            n = np.zeros_like(v)
            c = np.zeros((N, 4))
            v[:, :2] = mask.nonzero()
            n[:, 2] = 1.

            # We'll use a voxel grid with 128 voxels per axis
            num_voxels_per_axis = 128

            # Size of the axis aligned bounding box of the point cloud
            bbox_size = v.max(0) - v.min(0) + np.array([0., 0., 0.1])

            # The size per-axis of a single voxel
            sizeof_voxel = bbox_size / num_voxels_per_axis

            # Downsample a point cloud on a voxel grid so there is at most one point per voxel.
            # Multiple points, normals, and colors within a voxel cell are averaged together.
            v_sampled, n_sampled, c_sampled = pcu.downsample_point_cloud_voxel_grid(sizeof_voxel, v, n, c)
            mask_idx_spl = v_sampled[:, :2]
            mask_spl = torch.zeros_like(mask)
            mask_spl[mask_idx_spl[:, 0], mask_idx_spl[:, 1]] = True
            self.mask = mask_spl
            print("After subsample:", mask_spl.shape[0])
            if self.cuda:
                self.mask = mask_spl.cuda()
        else:
            self.mask = mask
            if self.cuda:
                self.mask = mask.cuda()
        self.rgb_np = rgb_np
        
    def set_init(self, q):
        self.q = q
        if self.cuda:
            self.q = q.cuda()
        
    def dist_fn(self, q):
        sigma = 3
        c = q_to_c_2d(q, self.H, self.W, cuda=self.cuda)
        xy = construct_xy(self.H, self.W, cuda=self.cuda)
        prev_dist = dist_to_line_2d(xy.clone(), c[0], c[1])
        for i in range(c.shape[0]-1):
            dist = torch.minimum(dist_to_line_2d(xy.clone(), c[i], c[i+1]), prev_dist)
            prev_dist = dist
#         dist = torch.exp(-dist * dist/(sigma**2))
        dist = dist[self.mask]
        return dist
        
    def step(self):
        # fwd
        prev_q = self.q.clone()
        changed = False
        step_size = 0.5/self.mask.sum()
        weight_decay = 0.9
        step = 0
        print('---- optimize translation ----')
        while((((prev_q-self.q).norm().item() > 0.5) or not changed) and step < 10):
            prev_q = self.q.clone()
            q_grad = self.q.clone().detach().requires_grad_(True)
            loss = loss_fn(q_grad, H, W, mask)
            print('loss:', loss)
            loss.backward()
            grad = q_grad.grad.clone().detach()
            grad[2:] = 0
            self.q = q_grad.clone().detach()
            self.q -= grad * step_size
            step_size *= weight_decay
            print(self.q)
            print(grad)
            changed = True
            step += 1

        changed = False
        step_size = 10**-5
        weight_decay = 0.9
        step = 0
        print('---- optimize rotation ----')
        while((((prev_q-self.q).norm().item() > 0.5) or not changed) and step < 10):
            prev_q = self.q.clone()
            q_grad = self.q.clone().detach().requires_grad_(True)
            loss = loss_fn(q_grad, H, W, mask)
            loss.backward()
            grad = q_grad.grad.clone().detach()
            self.q = q_grad.clone().detach()
            self.q -= grad * step_size
            step_size *= weight_decay
            print(self.q)
            print(grad)
            changed = True
            step += 1
    
    def gauss_step(self):
#         print('---- optimize translation ----')
#         curr_norm = float('Inf')
#         while True:
#             prev_norm = curr_norm
#             J=jacobian(self.dist_fn, self.q)
#             dist = self.dist_fn(self.q)
#             curr_norm = dist.norm()
#             print(self.q)
#             print(dist.norm())
#             if curr_norm >= prev_norm:
#                 self.q = prev_q
#                 break
#             prev_q = self.q.clone()
#             try:
#                 delta_q = torch.mm(torch.mm(torch.mm(J.t(), J).inverse(), J.t()), dist.unsqueeze(1)).squeeze()
#                 delta_q[2:] = 0
#             except:
#                 delta_q = torch.mm(torch.mm(torch.mm(J.t(), J).pinverse(), J.t()), dist.unsqueeze(1)).squeeze()
#                 delta_q[2:] = 0
#             self.q -= delta_q
        
        print('---- optimize rotation ----')
        curr_norm = float('Inf')
        while True:
            prev_norm = curr_norm
            J=jacobian(self.dist_fn, self.q)
            dist = self.dist_fn(self.q)
            curr_norm = dist.norm()
            print(self.q)
            print(dist.norm())
            if curr_norm >= prev_norm:
                self.q = prev_q
                break
            prev_q = self.q.clone()
            try:
                delta_q = torch.mm(torch.mm(torch.mm(J.t(), J).inverse(), J.t()), dist.unsqueeze(1)).squeeze()
            except:
                delta_q = torch.mm(torch.mm(torch.mm(J.t(), J).pinverse(), J.t()), dist.unsqueeze(1)).squeeze()
            self.q -= delta_q
    
    def lm_step(self):
        pass
            
    def vis(self, save_dir=None, idx=0):
        if self.rgb_np is not None:
            vis_img = self.rgb_np.copy()
        else:
            vis_img = np.zeros((self.H, self.W))

        radius = 3
        color = (255, 0, 0)
        thickness = 2

        c = q_to_c_2d(self.q, self.H, self.W)
        for i in range(c.shape[0]):
            vis_img = cv2.circle(vis_img, (int(c[i][0]), self.H-int(c[i][1])), radius, color, thickness)
        plt.imshow(vis_img)
        if save_dir is not None:
            import pathlib
            pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
            plt.savefig(save_dir+"frame_"+'{0:03d}'.format(idx)+".png")
        plt.show()
```

```python
# gradient descent sanity check
H = 540
W = 810
simple_tracker = tracker(H, W,cuda=True)
q = torch.zeros((12,))
q[0] = 180
q[1] = 360
simple_tracker.set_init(q)
mask = torch.zeros((H, W), dtype=bool)
mask[200:205, 200:300] = True
simple_tracker.set_obs(mask)
simple_tracker.step()
```

```python
# gauss-newton method sanity check
H = 540
W = 810
simple_tracker = tracker(H, W,cuda=True)
q = torch.zeros((12,))
q[0] = 180
q[1] = 360
simple_tracker.set_init(q)
mask = torch.zeros((H, W), dtype=bool)
mask[200:205, 200:300] = True
simple_tracker.set_obs(mask)
simple_tracker.gauss_step()
```

```python
simple_tracker.vis()
```

```python
# simple rope tracking
H = 540
W = 810
simple_tracker = tracker(H, W, cuda=True)
q = torch.zeros((20,))
q[0] = 40
simple_tracker.set_init(q)
data_path = "/home/yixuan/dart_deformable/data/rope_simple/"
for i in range(251):
    rgb_np = exr_to_np(data_path+"rgb_"+'{0:03d}'.format(i)+".exr")
    mask = rgb_np[:, :, 2] > 100
    simple_tracker.set_obs(torch.tensor(mask), rgb_np)
    simple_tracker.gauss_step()
    simple_tracker.vis(save_dir="/home/yixuan/dart_deformable/result/rope_simple/", idx=i)
```

```python
# simple rope tracking with subsample
H = 540
W = 810
simple_tracker_sub = tracker(H, W, cuda=True)
# simple_tracker = tracker(H, W, cuda=True)
q = torch.zeros((20,))
q[0] = 40
# simple_tracker.set_init(q)
simple_tracker_sub.set_init(q)
data_path = "/home/yixuan/dart_deformable/data/rope_simple/"
for i in range(251):
    rgb_np = exr_to_np(data_path+"rgb_"+'{0:03d}'.format(i)+".exr")
    mask = rgb_np[:, :, 2] > 100
#     simple_tracker.set_obs(torch.tensor(mask), rgb_np, subsample=False)
    simple_tracker_sub.set_obs(torch.tensor(mask), rgb_np, subsample=True)
    start = time.time()
#     simple_tracker.gauss_step()
#     print("without subsample:", time.time()-start)
    start = time.time()
    simple_tracker_sub.gauss_step()
    print("with subsample:", time.time()-start)
    simple_tracker_sub.vis(save_dir="/home/yixuan/dart_deformable/result/rope_simple_sub/", idx=i)
```

```python

```
