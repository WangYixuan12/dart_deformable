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
import skfmm
from scipy.optimize import minimize
```

```python
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
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
def depth_exr_to_np(path):
    y_resolution = 540
    x_resolution = 810
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    
    depth_exr = OpenEXR.InputFile(path)
    depth_dw = depth_exr.header()['dataWindow']
    depth_size = (depth_dw.max.x - depth_dw.min.x + 1, depth_dw.max.y - depth_dw.min.y + 1)
    depth_str = depth_exr.channel('R', pt)
    depth = Image.frombytes("F", depth_size, depth_str)
    depth_img = np.array(depth.getdata()).reshape(y_resolution, x_resolution)
    depth_img = depth_img * 1000.0
    return depth_img
```

```python
def threshold(img, v_min, v_max):
    '''
    threshold an img using v_min and v_max; img.shape[2] = v_min.shape[0] = v_max.shape[0]
    '''
    H, W, C = img.shape
    mask =  np.ones((H, W))
    for i in range(C):
        mask_i_1 = img[:, :, i] > v_min[i]
        mask_i_2 = img[:, :, i] < v_max[i]
        mask_i = np.logical_and(mask_i_1, mask_i_2)
        mask = np.logical_and(mask, mask_i)
    return mask
```

```python
class tracker:
    def __init__(self, H, W, cuda=False):
        self.H = H
        self.W = W
        self.cuda = cuda
        if cuda == True:
            self.device='cuda'
        else:
            self.device='cpu'
        
    def set_obs(self, mask, rgb_np=None, depth_np=None, subsample=False):
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
            plt.imshow(mask)
            plt.show()
            mask = (~mask).astype(np.uint8)
            phi = np.where(mask, 0, -1) + 0.5
            dist = skfmm.distance(phi, dx = 1)
            plt.imshow(dist)
            plt.show()
            dist = dist - dist.min() - 1
#             dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
            sobelx = cv2.Sobel(dist,cv2.CV_64F,1,0,ksize=-1)/30.56
            sobely = cv2.Sobel(dist,cv2.CV_64F,0,1,ksize=-1)/30.56
            
            self.mask = torch.Tensor(mask)
            self.dist = torch.Tensor(dist)
            self.dist_x = torch.Tensor(sobelx)
            self.dist_y = torch.Tensor(sobely)
            if self.cuda:
                self.mask = self.mask.cuda()
                self.dist = self.dist.cuda()
                self.dist_x = self.dist_x.cuda()
                self.dist_y = self.dist_y.cuda()
                
        self.rgb_np = rgb_np
        self.depth_np = depth_np
        
    def set_init(self, q, root_idx=0):
        self.q = q
        self.root_idx = 
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
    
    def q_to_c_2d(self, q):
        l = 20
        n = q.shape[0]-1
        c = torch.zeros((n,2), device=q.device)
        c[0, 0] = q[0]
        c[0, 1] = q[1]
        for i in range(1, n):
            c[i, 0] = c[i-1, 0] + l*torch.cos(q[i+1])
            c[i, 1] = c[i-1, 1] + l*torch.sin(q[i+1])
    #     print(c)
        c_in_img = c.clone()
        c_in_img[:, 0] = self.H-c[:, 1]-1
        c_in_img[:, 1] = c[:, 0]
        return c_in_img

    def q_to_c_2d_flat(self, q):
        return self.q_to_c_2d(q).view(-1)
    
    def gauss_obj_step_trans(self):
        # translation
        curr_norm = float('Inf')
        lam = 10.0
        last_q = self.q.clone()
        while True:
            prev_norm = curr_norm
            nabla_q=jacobian(self.q_to_c_2d_flat, self.q) # (2*Nc, Nq)
            c = self.q_to_c_2d(self.q) # (Nc, 2)
            nabla_D = torch.zeros_like(c, device=c.device) # (Nc, 2)
            c = c.long()
            
            # clip within image
            c = torch.maximum(c, torch.zeros_like(c, device=c.device))
            c[:, 0] = torch.minimum(c[:, 0], (self.H-1)*torch.ones_like(c[:, 0], device=c.device))
            c[:, 1] = torch.minimum(c[:, 1], (self.W-1)*torch.ones_like(c[:, 1], device=c.device))
            
            nabla_D[:, 1] = self.dist_x[c[:, 0], c[:, 1]]
            nabla_D[:, 0] = self.dist_y[c[:, 0], c[:, 1]]
            nabla_D = nabla_D.view(-1) # (2*Nc, )
            nabla_D = torch.diag(nabla_D) # (2*Nc, 2*Nc)
            J = torch.mm(nabla_D, nabla_q) # (2*Nc, Nq)
            J = J[0::2]+J[1::2] # (Nc, Nq)
            
            # regularize
            J_reg = lam*torch.eye(self.q.shape[0], device = J.device)
            J = torch.vstack((J, J_reg[2:]))
            
            # calculate loss
            dist = self.dist[c[:, 0], c[:, 1]]
            dist = torch.hstack((dist, lam*(self.q-last_q)[2:]))
            curr_norm = torch.square(dist).sum()
            
            print("q:")
            print(self.q)
            print("loss:")
            print(curr_norm)
#             print("dist:")
#             print(dist)
#             print("nabla_q:")
#             print(nabla_q)
#             print("nabla_D:")
#             print(nabla_D)
#             print("J:")
#             print(J)
#             print("Gradient:")
#             print(torch.mm(J.t(), dist.unsqueeze(1)))
            
            if curr_norm >= prev_norm:
                self.q = prev_q
                break
            prev_q = self.q.clone()
            try:
                delta_q = torch.mm(torch.mm(torch.mm(J.t(), J).inverse(), J.t()), dist.unsqueeze(1)).squeeze()
            except:
                delta_q = torch.mm(torch.mm(torch.mm(J.t(), J).pinverse(), J.t()), dist.unsqueeze(1)).squeeze()
            if delta_q.norm() < 1e-6:
                break
            delta_q[2:] = 0.
            self.q -= delta_q
    
    def gauss_obj_step(self, debug=False):
        # rotation
        curr_norm = float('Inf')
        lam = 10.0
        last_q = self.q.clone()
        while True:
            prev_norm = curr_norm
            nabla_q=jacobian(self.q_to_c_2d_flat, self.q) # (2*Nc, Nq)
            c = self.q_to_c_2d(self.q) # (Nc, 2)
            nabla_D = torch.zeros_like(c, device=c.device) # (Nc, 2)
            c = c.long()
            
            # clip within image
            c = torch.maximum(c, torch.zeros_like(c, device=c.device))
            c[:, 0] = torch.minimum(c[:, 0], (self.H-1)*torch.ones_like(c[:, 0], device=c.device))
            c[:, 1] = torch.minimum(c[:, 1], (self.W-1)*torch.ones_like(c[:, 1], device=c.device))
            
            nabla_D[:, 1] = self.dist_x[c[:, 0], c[:, 1]]
            nabla_D[:, 0] = self.dist_y[c[:, 0], c[:, 1]]
            nabla_D = nabla_D.view(-1) # (2*Nc, )
            nabla_D = torch.diag(nabla_D) # (2*Nc, 2*Nc)
            J = torch.mm(nabla_D, nabla_q) # (2*Nc, Nq)
            J = J[0::2]+J[1::2] # (Nc, Nq)
            
            # regularize
            J_reg = lam*torch.eye(self.q.shape[0], device = J.device)
            J = torch.vstack((J, J_reg[2:]))
            
            # calculate loss
            dist = self.dist[c[:, 0], c[:, 1]]
            dist = torch.hstack((dist, lam*(self.q-last_q)[2:]))
            curr_norm = torch.square(dist).sum()
            
            if debug:
                print("q:")
                print(self.q)
                print("loss:")
                print(curr_norm)
                print("dist:")
                print(dist)
                print("nabla_q:")
                print(nabla_q)
                print("nabla_D:")
                print(nabla_D)
                print("J:")
                print(J)
                print("Gradient:")
                print(torch.mm(J.t(), dist.unsqueeze(1)))
            
            if curr_norm >= prev_norm:
                self.q = prev_q
                break
            prev_q = self.q.clone()
            try:
                delta_q = torch.mm(torch.mm(torch.mm(J.t(), J).inverse(), J.t()), dist.unsqueeze(1)).squeeze()
            except:
                delta_q = torch.mm(torch.mm(torch.mm(J.t(), J).pinverse(), J.t()), dist.unsqueeze(1)).squeeze()
            if delta_q.norm() < 1e-6:
                break
            self.q -= delta_q
            
    def occl_weight(self, c):
        '''
        Input:
            c: (Nc, 2) torch tensor, pixel of the image
        Output:
            w: (Nc,) torch tensor, weight of every pixel
        '''
        Nc, _ = c.shape
        occl_map = self.depth_np < 2000.0
        dist_map = torch.Tensor(skfmm.distance(~occl_map, dx = 1)).to(self.device)
        dist = dist_map[c[:, 0], c[:, 1]]
        sigma = 30.
        w = 1-torch.exp(-dist/sigma)
        w[dist>sigma] = 1.0
        return w
    
    def gauss_obj_occl_step(self):
        # Idea: occluded points are not counted in the objective function
        curr_norm = float('Inf')
        lam = 10.0
        last_q = self.q.clone()
        occl_map = torch.BoolTensor(self.depth_np < 2000.0).to(self.device)
        while True:
            prev_norm = curr_norm
            c = self.q_to_c_2d(self.q) # (Nc, 2)
            c = c.long()
            
            # clip within image
            c = torch.maximum(c, torch.zeros_like(c, device=c.device))
            c[:, 0] = torch.minimum(c[:, 0], (self.H-1)*torch.ones_like(c[:, 0], device=c.device))
            c[:, 1] = torch.minimum(c[:, 1], (self.W-1)*torch.ones_like(c[:, 1], device=c.device))
            
            w = self.occl_weight(c)
            
            nabla_q=jacobian(self.q_to_c_2d_flat, self.q) # (2*Nc, Nq)
            
            # exclude those occluded
#             c_idx = (~occl_map[c[:, 0], c[:, 1]]).nonzero().reshape(-1)
#             print("points not occluded")
#             print(c_idx)
#             c = c[c_idx]
#             Nc_2, Nq = nabla_q.shape
#             Nc = Nc_2//2
#             nabla_q = nabla_q.reshape(2, Nc, Nq)
#             nabla_q = nabla_q[:, c_idx, :].reshape(-1, Nq)
            
            nabla_D = torch.zeros_like(c, device=c.device).float() # (Nc, 2)
            nabla_D[:, 1] = self.dist_x[c[:, 0], c[:, 1]]
            nabla_D[:, 0] = self.dist_y[c[:, 0], c[:, 1]]
            nabla_D = nabla_D.reshape(-1) # (2*Nc, )
            nabla_D = torch.diag(nabla_D) # (2*Nc, 2*Nc)
            J = torch.mm(nabla_D, nabla_q) # (2*Nc, Nq)
            J = J[0::2]+J[1::2] # (Nc, Nq)
            J = J*w[:, None]
            
            # regularize
            J_reg = lam*torch.eye(self.q.shape[0], device = J.device)
            J = torch.vstack((J, J_reg[2:]))
            
            # calculate loss
            dist = self.dist[c[:, 0], c[:, 1]]
            dist = dist*w
            dist = torch.hstack((dist, lam*(self.q-last_q)[2:]))
            curr_norm = torch.square(dist).sum()
            
            print("q:")
            print(self.q)
            print("loss:")
            print(curr_norm)
#             print("dist:")
#             print(dist)
#             print("nabla_q:")
#             print(nabla_q)
#             print("nabla_D:")
#             print(nabla_D)
#             print("J:")
#             print(J)
#             print("Gradient:")
#             print(torch.mm(J.t(), dist.unsqueeze(1)))
            
            if curr_norm >= prev_norm:
                self.q = prev_q
                break
            prev_q = self.q.clone()
            try:
                delta_q = torch.mm(torch.mm(torch.mm(J.t(), J).inverse(), J.t()), dist.unsqueeze(1)).squeeze()
            except:
                delta_q = torch.mm(torch.mm(torch.mm(J.t(), J).pinverse(), J.t()), dist.unsqueeze(1)).squeeze()
            if delta_q.norm() < 1e-6:
                break
            self.q -= delta_q
        
    def vis(self, save_dir=None, idx=0):
        if self.rgb_np is not None:
            vis_img = self.rgb_np.copy()
        else:
            vis_img = np.zeros((self.H, self.W))

        radius = 5
        color = (255, 0, 0)
        thickness = 3

        c = self.q_to_c_2d(self.q)
        for i in range(c.shape[0]):
            vis_img = cv2.circle(vis_img, (int(c[i][1]), int(c[i][0])), radius, color, thickness)
        plt.imshow(vis_img)
        if save_dir is not None:
            import pathlib
            pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
            plt.savefig(save_dir+"frame_"+'{0:03d}'.format(idx)+".png", dpi=300)
        plt.show()
```

```python
# distance transform
data_path = "/home/yixuan/dart_deformable/data/rope_simple/"
i = 0
rgb_np = exr_to_np(data_path+"rgb_"+'{0:03d}'.format(i)+".exr")
# _, mask = cv2.threshold(rgb_np[:, :, 2], 100, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
mask = rgb_np[:, :, 2] > 100
mask = (~mask).astype(np.uint8)
mask[200:300, 10:500] = 0
# print(mask.shape)
# print(mask.dtype)
# dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
phi = np.where(mask, 0, -1) + 0.5
dist = skfmm.distance(phi, dx = 1)
plt.imshow(dist)
plt.colorbar()
plt.show()
sobelx = cv2.Sobel(dist,cv2.CV_64F,1,0,ksize=-1)
sobely = cv2.Sobel(dist,cv2.CV_64F,0,1,ksize=-1)
plt.imshow(sobelx)
plt.show()
plt.imshow(sobely)
plt.show()
```

```python
# gauss-newton method sanity check
H = 540
W = 810
simple_tracker = tracker(H, W,cuda=True)
q = torch.zeros((12,))
q[0] = 180
q[1] = 10
simple_tracker.set_init(q)
mask = np.zeros((H, W), dtype=bool)
mask[200:205, 200:400] = True
simple_tracker.set_obs(mask)
simple_tracker.gauss_obj_step()
simple_tracker.vis()
```

```python
# simple rope tracking
H = 540
W = 810
simple_tracker_sub = tracker(H, W, cuda=True)
q = torch.zeros((20,))
q[0] = 40
q[1] = 10
simple_tracker_sub.set_init(q)
data_path = "/home/yixuan/dart_deformable/data/rope_simple/"
for i in range(251):
    rgb_np = exr_to_np(data_path+"rgb_"+'{0:03d}'.format(i)+".exr")
    mask = rgb_np[:, :, 2] > 100
    simple_tracker_sub.set_obs(mask, rgb_np)
    start = time.time()
    simple_tracker_sub.gauss_obj_step()
    print("one iteration takes:", time.time()-start)
    simple_tracker_sub.vis(save_dir="/home/yixuan/dart_deformable/result/rope_simple_obj_loss/", idx=i)
```

```python
# simple occluded rope tracking
H = 540
W = 810
occl_tracker = tracker(H, W, cuda=True)
q = torch.zeros((20,))
q[0] = 40
q[1] = 10
occl_tracker.set_init(q)
data_path = "/home/yixuan/blender_data/rope_simple_occlusion/render/"
for i in range(251):
    rgb_np = exr_to_np(data_path+"rgb_"+'{0:03d}'.format(i)+".exr")
    depth_np = depth_exr_to_np(data_path+"depth_"+'{0:03d}'.format(i)+".exr")
    hsv_np = cv2.cvtColor(rgb_np, cv2.COLOR_BGR2HSV)
    mask = np.bitwise_and(hsv_np[:, :, 1] > 150, hsv_np[:, :, 0] < 30)
    occl_tracker.set_obs(mask, rgb_np, depth_np)
    start = time.time()
    occl_tracker.gauss_obj_occl_step()
    print("one iteration takes:", time.time()-start)
    occl_tracker.vis(save_dir="/home/yixuan/dart_deformable/result/rope_occl_obj_loss/", idx=i)
```

```python
# simple tracking in real data
H = 720
W = 1280
track = tracker(H, W, cuda=True)

start_idx = 414
# 1675 - 2022_01_05_14_35_00
end_idx = 842
# 2285 - 2022_01_05_14_35_00

q = torch.zeros((21,))
q[0] = 40
q[1] = 300
track.set_init(q)
data_path = "/home/yixuan/Downloads/rope_dataset_0105_no_marker/2022_01_05_17_59_17/2022_01_05_17_59_17/"
for i in range(start_idx, end_idx, 1):
    rgb_np = cv2.imread(data_path+"realsense_overhead_5_l515_color/"+str(i)+".jpg")
    depth_np = cv2.imread(data_path+"realsense_overhead_5_l515_depth/"+str(i)+".png", cv2.IMREAD_ANYDEPTH)
    hsv_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2HSV)
    
    # mask image
    depth_mask = np.logical_and(depth_np < 1000, depth_np > 600)
    
#     marker_min = np.array([20,0,200])
#     marker_max = np.array([100,100,300])
#     marker_mask = threshold(hsv_np, marker_min, marker_max)
#     marker_mask = np.logical_and(marker_mask, depth_mask)

    rope_min = np.array([0,75,200])
    rope_max = np.array([60,250,300])
    rope_mask = threshold(hsv_np, rope_min, rope_max)
    rope_mask = np.logical_and(rope_mask, depth_mask)
#     rope_mask = np.logical_or(rope_mask, marker_mask)
    
    track.set_obs(rope_mask, rgb_np, depth_np)
    start = time.time()
    if i == start_idx:
        track.gauss_obj_step_trans()
    if i >= 524 and i <= 542:
        track.gauss_obj_step(debug = True)
        print("one iteration takes:", time.time()-start)
        track.vis(save_dir="/home/yixuan/dart_deformable/result/rope_dataset_0105_no_occl/", idx=i)
    else:
        track.gauss_obj_step()
        print("one iteration takes:", time.time()-start)
```

```python
def imgs_to_video(path):
    rgb = cv2.imread(path+"rgb_000.png")
    H, W, _ = rgb.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(path+'dataset.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 20, (W, H))
    for i in range(251):
        rgb = cv2.imread(path+"rgb_"+'{0:03d}'.format(i)+".png")
        out.write(rgb)
    out.release()
```

```python
path="/home/yixuan/blender_data/rope_simple_occlusion/render/"
imgs_to_video(path)
```

```python
a=np.
```

```python

```
