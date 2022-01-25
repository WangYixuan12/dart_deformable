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
from scipy.optimize import minimize, Bounds
from autograd import grad, jacobian
from jax import jacfwd, jacrev, grad
import jax.numpy as jnp
import pathlib
import random
```

```python
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['JAX_DEBUG_NANS'] = '1'
```

```python
from jax.config import config
config.update("jax_debug_nans", True)
```

```python
def construct_xy(H, W, cuda=False):
    xy = np.zeros((2, H, W))
    xy[0] = np.arange(W).reshape(1, W).repeat((H, 1)) # x
    xy[1] = np.arange(H).reshape(H, 1).repeat((1, W)) # y
    return xy
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
def sign_nozero(p):
    '''
    Similar to np.sign but output is 1 if value is zero
    '''
    p = p.astype(float)
    s = np.sign(p)
    s[s == 0] = 1
    return s
```

```python
def dist_to_line_2d(xy, c1, c2):
    '''
    Input:
        xy: (2/3, Nm) jax numpy array - (x, y, [z]) is the coordinate
        c1: (2/3,) jax numpy array - (x,y,[z]) of one end
        c2: (2/3,) jax numpy array - (x,y,[z]) of another end
    '''
    xy_c1 = xy - c1.reshape(-1,1)
    c12 = (c2 - c1).reshape(-1,1)
    c12_length = jnp.linalg.norm(c12)
    c12 = c12/c12_length
    closest_dist = jnp.clip(jnp.sum(xy_c1 * c12, axis=0), 0, c12_length)
    closest_pt = c1.reshape(-1,1) + closest_dist * c12
#     dist = jnp.linalg.norm(xy - closest_pt+1e-10, axis = 0)
    dist = jnp.sum(jnp.square(xy - closest_pt), axis = 0)
    # otherwise derivative at 0 is undefined
    return dist
```

```python
class indi_pt_tracker:
    def __init__(self, H, W, l, debug=False, reg=False, ratio=1.0, lam=100., dim=2, cam_K=None):
        self.H = H
        self.W = W
        self.l = l
        self.lam = lam
        self.debug = debug
        self.reg = reg
        self.ratio = ratio
        self.model_jac_fn_jax = jacfwd(self.model_obj_fn)
        self.dim = dim
        self.cam_K = cam_K
        
    def set_obs(self, mask, rgb_np=None, depth_np=None, subsample=False):
        if self.dim == 2:
            if subsample:
                import point_cloud_utils as pcu

                # v is a nv by 3 NumPy array of vertices
                # n is a nv by 3 NumPy array of vertex normals
                # n is a nv by 4 NumPy array of vertex colors
                N = np.array(mask.nonzero()).shape[1]
                v = np.zeros((N, 3))
                n = np.zeros_like(v)
                c = np.zeros((N, 4))
                v[:, :2] = np.array(mask.nonzero()).T
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
                mask_idx_spl = v_sampled[:, :2].astype(int)
                mask_spl = np.zeros_like(mask)
                mask_spl[mask_idx_spl[:, 0], mask_idx_spl[:, 1]] = True
                self.mask = mask_spl

            # occlusion weight map
            if depth_np is not None:
                occl_map = depth_np < 2000.0
                occl_dist = skfmm.distance(~occl_map, dx = 1)
                sigma = 30.
    #             self.occl_w_map = 1-np.exp(-occl_dist/sigma)
                self.occl_w_map = np.zeros_like(occl_dist)
                self.occl_w_map[occl_dist<=sigma] = 0.0
                self.occl_w_map[occl_dist>sigma] = 1.0
            else:
                self.occl_w_map = np.ones_like(mask).astype(float)

            mask = (~mask).astype(np.uint8)
            phi = np.where(mask, 0, -1) + 0.5
            dist = skfmm.distance(phi, dx = 1)
            dist = dist - dist.min() # - 1
            sobelx = cv2.Sobel(dist,cv2.CV_64F,1,0,ksize=-1)/30.56
            sobely = cv2.Sobel(dist,cv2.CV_64F,0,1,ksize=-1)/30.56

            self.full_mask = mask
            self.dist = dist
            self.dist_x = sobelx
            self.dist_y = sobely
            
        elif self.dim == 3:
            # RGBD to pc
            unit = 1000.0
            img = img/unit
            mask_idx=mask.nonzero()
            X = (mask_idx[1]-cx)*img/fx
            Y = (mask_idx[0]-cy)*img/fy
            pc = np.zeros((mask_idx[0].shape[0], 3))
            pc[:,0]=X.reshape(-1)
            pc[:,1]=Y.reshape(-1)
            pc[:,2]=depth_np[mask].reshape(-1)
            self.pc = pc # Nm, 3
        
        self.rgb_np = rgb_np
        self.depth_np = depth_np
    
    def find_nonmask(self, p):
        '''
        Input: p: jax numpy array (Np, 2)
        '''
        nonmask_idx = jnp.array(self.full_mask.nonzero())
        Np = p.shape[0]
        dist_all = jnp.zeros((Np-1,nonmask_idx.shape[1]))
        for i in range(Np-1):
            dist_all = dist_all.at[i].set(dist_to_line_2d(nonmask_idx, p[i], p[i+1]))
        dist = dist_all.min(axis=0)
        nonmask_idx = nonmask_idx[:, dist < 10]
        return nonmask_idx
    
    def p2ang_2d(self, p):
        '''
        Input: p: numpy array (Np, 2)
        Output: ang: numpy array (Ne)
        '''
        p = p.astype(float)
        dx = p[1:, 1] - p[:-1, 1]
        dy = p[1:, 0] - p[:-1, 0]
        ang_0_pi = np.arccos(dx/np.sqrt(np.square(dx)+np.square(dy)))
        dy[ang_0_pi==0] += 0.01 # singular value for ang_0_pi = 0 or anf_0_pi = pi
        ang = np.sign(dy)*ang_0_pi
        return ang
    
    def p2dang(self, p):
        '''
        Input: p: numpy array (Np, 3)
        Output: dg numpy array (Ne)
        '''
        p = p.astype(float)
        l23 = p[2:] - p[1:-1]
        l12 = p[1:-1] - p[0:-2]
        num = np.sum(l12*l23, axis=0)
        den = np.linalg.norm(l12, axis=0) * np.linalg.norm(l23, axis=0)
        dang = np.arccos(num/den)
        return dang
    
    def hard_constrain(self, p):
        '''
        p: numpy array (Np,2)
        '''
        p = np.maximum(p, np.zeros_like(p))
        p[:,0] = np.minimum(p[:,0], np.ones(p[:,0].shape, dtype=int)*(self.H-1))
        p[:,1] = np.minimum(p[:,1], np.ones(p[:,1].shape, dtype=int)*(self.W-1))
        return p
    
    def set_init(self, p):
        '''
        p: numpy array (Np,2)
        '''
        self.p = p
        self.ang = self.p2ang_2d(self.p)
        lb = np.zeros(p.shape[0]*2)
        ub = np.ones(p.shape[0]*2)
        ub[0::2] = self.H-1
        ub[1::2] = self.W-1
        self.bound = Bounds(lb, ub)
    
    '''
    OBJECTIVE FUNCTION DEFINITION
    '''
    
    def obj_fn(self, p):
        '''
        Input: p: numpy array, (2*Np,)
        '''
#         ratio = np.array(self.mask.nonzero()).shape[1]/(5.0*p.shape[0])
        if self.debug:
            print('---- In Obj Fn ----')
        ratio = self.ratio
        l_obs = self.obs_obj_fn(p)
        p = jnp.array(p.reshape(-1, self.dim))
        l_model = self.model_obj_fn(p)
        loss = l_obs+ratio*l_model
        if self.debug:
            print('p:')
            self.vis(p=np.array(p).astype(int))
            print('l_obs:')
            print(l_obs)
            print('l_model:')
            print(l_model)
            print('loss')
            print(loss)
            print('---- End of Obj Fn ----')
        return loss
    
    def obs_obj_fn(self, p):
        '''
        p: numpy array (2*Np,)
        '''
        loss = 0
        p = p.reshape(-1,self.dim)
        # TODO: to impl reg term in 3d
        if self.reg:
#             d_ang = self.p2ang_2d(p) - self.ang
#             loss += np.sum(self.lam*np.abs(d_ang))
            ang = self.p2ang_2d(p)
            loss += 0.5*self.lam*np.sum(np.square(ang[:-1]-ang[1:]))
        p = p.astype(int)
        p = self.hard_constrain(p)
        loss += 0.5*np.sum( np.square(self.occl_w_obs*self.dist[p[:,0], p[:,1]]) )
#         loss += np.sum(self.occl_w_obs[:,None]*self.dist[p[:,0], p[:,1]])
        
        if self.debug:
            print('distance:')
            print(self.occl_w_obs*self.dist[p[:,0], p[:,1]])
            print('square distance:')
            a = np.square(self.occl_w_obs*self.dist[p[:,0], p[:,1]])
            print(a)
            print('loss of distance:')
            print(0.5*np.sum( np.square(self.occl_w_obs*self.dist[p[:,0], p[:,1]]) ))
            if self.reg:
                print('ang:')
                print(ang)
                print('loss of angle:')
                print(0.5*self.lam*np.sum(np.square(ang[:-1]-ang[1:])))
            print('obs value:')
            print(loss)
        return loss
    
    def model_obj_fn(self, p):
        '''
        Input: p: jax numpy array, (Np,2)
        '''
        if self.dim == 2:
            mask_idx = jnp.array(self.mask.nonzero())
    #         nonmask_idx = self.find_nonmask(p)
            Np = p.shape[0]
            dist_mask = jnp.zeros((Np-1,mask_idx.shape[1]))
    #         dist_nonmask = jnp.zeros((Np-1,nonmask_idx.shape[1]))
    #         prev_dist = dist_to_line_2d(mask_idx, p[0], p[1])
            for i in range(Np-1):
                # TODO: add occlusion weight here
    #             dist_all = dist_all.at[i].set(dist_to_line_2d(mask_idx, p[i], p[i+1])/(self.occl_w[i]+0.01))
                dist_mask = dist_mask.at[i].set(dist_to_line_2d(mask_idx, p[i], p[i+1]))
    #             if self.occl_w_obs[i] < 0.5:
    #                 dist_mask_to_pi = float('Inf')*jnp.ones(mask_idx.shape[1])
    #             else:
    #                 dist_mask_to_pi = jnp.linalg.norm(mask_idx-p[i][:, None]+1e-10, axis=0)/(self.occl_w_obs[i]+1e-10)
    #             dist_mask = dist_mask.at[i].set(dist_mask_to_pi)
    #             print(jnp.linalg.norm(mask_idx-p[i]+1e-10, axis=0).shape)
    #             print(mask_idx.shape)
    #             print(p[i].shape)
    #             dist_nonmask = dist_nonmask.at[i].set(dist_to_line_2d(nonmask_idx, p[i], p[i+1]))
    #             dist = jnp.minimum(dist_to_line_2d(mask_idx, p[i], p[i+1]), prev_dist)
    #             prev_dist = dist
    #         weight_seg_idx = (dist_mask/(self.occl_w_obs+1e-10)[:, None]).argmin(axis=0)
    #         seg_idx = dist_mask.argmin(axis=0)
    #         print(jnp.abs(weight_seg_idx-seg_idx).sum())
    #         weight_dist = dist_mask[seg_idx, np.arange(mask_idx.shape[1])]
            dist = dist_mask.min(axis=0)
    #         print(jnp.abs(weight_dist-dist).sum())
    #         nondist = dist_nonmask.min(axis=0)
    #         occl_w = jnp.array(self.occl_w)
    #         dist = dist*occl_w[seg_idx]
        else:
            pc_jnp = jnp.array(self.pc.T)
            Np = p.shape[0]
            dist_mask = jnp.zeros((Np-1,pc_jnp.shape[0]))
            for i in range(Np-1):
                dist_mask = dist_mask.at[i].set(dist_to_line_2d(pc_jnp, p[i], p[i+1]))
            dist = dist_mask.min(axis=0)
        return dist.sum()#-nondist.sum()
    
    '''
    JACOBIAN FUNCTION DEFINITION
    '''
    
    def jac_fn(self, p):
        '''
        Input: p: numpy array, (2*Np,)
        '''
#         ratio = np.array(self.mask.nonzero()).shape[1]/(5.0*p.shape[0])
        if self.debug:
            print('---- In Jac Fn ----')
        ratio = self.ratio
        J_obs = self.obs_jac_fn(p)
        p = jnp.array(p.reshape(-1, 2))
        J_model = np.array(self.model_jac_fn_jax(p)).reshape(-1)
        J = J_obs + ratio*J_model
        if self.debug:
            print('p:')
            self.vis(p=np.array(p).astype(int))
            print('J:')
            print(J)
            print('J_obs:')
            print(J_obs)
            print('J_model:')
            print(J_model)
            print('---- End of Jac Fn ----')
        return J
    
    
    def obs_jac_fn(self, p):
        '''
        p: numpy array (2*Np,)
        '''
        p = p.reshape(-1,2)
        
        # Jacobian for the angle change
        J_ang = np.zeros(p.shape)
        if self.reg:
            dx = p[1:, 1] - p[:-1, 1]
            dy = p[1:, 0] - p[:-1, 0]
            l = np.square(dx)+np.square(dy)
#             d_ang = self.p2ang_2d(p) - self.ang
#             J_ang[1:, 0] += self.lam*np.sign(d_ang)*dx/l
#             J_ang[1:, 1] += -self.lam*np.sign(d_ang)*dy/l
#             J_ang[:-1,0] += -self.lam*np.sign(d_ang)*dx/l
#             J_ang[:-1,1] += self.lam*np.sign(d_ang)*dy/l
            ang = self.p2ang_2d(p)
            d_ang = ang[1:] - ang[:-1]
            J_ang[0:-2, 0] += self.lam*d_ang*dx[0:-1]/l[0:-1]
            J_ang[0:-2, 1] += -self.lam*d_ang*dy[0:-1]/l[0:-1]
            J_ang[1:-1, 0] += -self.lam*d_ang*(dx[0:-1]/l[0:-1]+dx[1:]/l[1:])
            J_ang[1:-1, 1] += self.lam*d_ang*(dy[0:-1]/l[0:-1]+dy[1:]/l[1:])
            J_ang[2:, 0] += self.lam*d_ang*dx[1:]/l[1:]
            J_ang[2:, 1] += -self.lam*d_ang*dy[1:]/l[1:]
            
        p = p.astype(int)
        p = self.hard_constrain(p)
        # Jacobian for distance residuals
        J_D = np.zeros(p.shape)
        J_D[:, 1] = self.occl_w_obs*self.dist_x[p[:, 0], p[:, 1]]
        J_D[:, 0] = self.occl_w_obs*self.dist_y[p[:, 0], p[:, 1]]
        J_D = (self.dist[p[:, 0], p[:, 1]]*self.occl_w_obs)[:,None]*J_D
        
        if self.debug:
            print('Distance Jacobian:')
            print(J_D)
            print('Angle Jacobian:')
            print(J_ang)
        return ((J_D+J_ang).reshape(-1))
    
    '''
    TRANSLATION TEST [ABANDONED]
    '''
    
    def trans_obj_fn(self, xy, p):
        '''
        xy: numpy array (2,)
        p: numpy array (Np,2)
        '''
        p = p + xy
        p = p.astype(int)
        J = np.zeros_like(xy)
        loss = 0.5*np.sum(np.square(self.dist[p[:,0], p[:,1]]))
        return loss
    
    def trans_jac_fn(self, xy, p):
        '''
        xy: numpy array (2,)
        p: numpy array (Np,2)
        '''
        p = p + xy
        p = p.astype(int)
        J_D = np.zeros(p.shape)
        J_D[:, 1] = self.dist_x[p[:, 0], p[:, 1]]
        J_D[:, 0] = self.dist_y[p[:, 0], p[:, 1]]
        J_D = self.dist[p[:, 0], p[:, 1]][:, None]*J_D
        J = J_D.sum(axis = 0)
        return J
    
    '''
    CONSTRAINTS
    '''
    
    def fixed_len(self, p):
        '''
        p: numpy array (2*Np,)
        '''
        p = p.reshape(-1,2)
        return np.sum(np.abs(np.linalg.norm(p[1:] - p[:-1], axis=1) - self.l))
    
    def fixed_len_i(self, p, i):
        p = p.reshape(-1,2)
        return np.linalg.norm(p[i]-p[i+1])-self.l
    
    def fixed_len_jac(self, p):
        '''
        p: numpy array (2*Np,)
        '''
        p = p.reshape(-1,2)
        dp = p[1:] - p[:-1]
        dp_norm = np.linalg.norm(dp, axis=1)
        
        J = np.zeros_like(p)
        J[:-1, 0] += np.sign(dp_norm-self.l)*(p[:-1,0]-p[1:,0])/dp_norm
        J[:-1, 1] += np.sign(dp_norm-self.l)*(p[:-1,1]-p[1:,1])/dp_norm
        J[1:, 0] += np.sign(dp_norm-self.l)*(p[1:,0]-p[:-1,0])/dp_norm
        J[1:, 1] += np.sign(dp_norm-self.l)*(p[1:,1]-p[:-1,1])/dp_norm
        if (J==0).all():
            J[:,:] = 1e-6
        if self.debug:
            print('In constrain jac')
            print(J)
#         J = np.sign(dp_norm-self.l)[:, None] * J
        return J.reshape(-1)

    def fixed_len_i_jac(self, p, i):
        p = p.reshape(-1,2)
        J = np.zeros_like(p)
        norm = np.linalg.norm(p[i]-p[i+1])
        J[i, 0] = (p[i, 0]-p[i+1, 0])/norm
        J[i, 1] = (p[i, 1]-p[i+1, 1])/norm
        J[i+1, 0] = (p[i+1, 0]-p[i, 0])/norm
        J[i+1, 1] = (p[i+1, 1]-p[i, 1])/norm
        return J.reshape(-1)
    
    '''
    STEP FUNCTION
    '''
    
    def trans_step(self):
        p = self.p.copy()
        xy = np.zeros(2)
        res = minimize(self.trans_obj_fn, xy, jac=self.trans_jac_fn, method='BFGS', args=(p))
        self.p = p + res.x
    
    def step(self, debug=False):
        self.debug = debug
        p = self.p.copy().reshape(-1)
        cons = []
#         fixed_len = {
#             'type': 'eq',
#             'fun': self.fixed_len,
#             'jac': self.fixed_len_jac
#         }
        opt = {
            'maxiter': 5000,
            'disp': False,
            'ftol': 1.0
        }
        for i in range(p.shape[0]//2-1):
            fixed_len_i = {
                'type': 'eq',
                'fun': self.fixed_len_i,
                'jac': self.fixed_len_i_jac,
                'args': [i]
            }
            cons.append(fixed_len_i)
        p_mean = (0.5*(self.p[:-1] + self.p[1:])).astype(int)
        self.occl_w = self.occl_w_map[p_mean[:, 0], p_mean[:, 1]]
        self.occl_w_obs = self.occl_w_map[self.p[:, 0].astype(int), self.p[:, 1].astype(int)]
        # TEST ONLY
#         self.occl_w_obs = np.ones(self.p.shape[0])
#         self.occl_w_obs[-1] = 0
#         self.occl_w_obs[-4:] = 0
#         print("model-based occlusion weight:")
#         print(self.occl_w)
#         print("observation-based occlusion weight:")
#         print(self.occl_w_obs)
#         print(p)
        res = minimize(self.obj_fn, p, jac=self.jac_fn, constraints=cons, method='SLSQP', bounds=self.bound, options=opt)
#         print(res)
        self.p = res.x.reshape(-1, 2)
        self.ang = self.p2ang_2d(self.p)
    
    def robust_step(self, debug=False):
        self.debug = debug
        self.full_mask_robust = ~(self.full_mask.copy().astype(bool)) # remember the full mask, which changed later
        self.init_p = self.p.copy()
        p_mean = (0.5*(self.p[:-1] + self.p[1:])).astype(int)
        self.occl_w = self.occl_w_map[p_mean[:, 0], p_mean[:, 1]]
        self.occl_w_obs = self.occl_w_map[self.p[:, 0].astype(int), self.p[:, 1].astype(int)]
        delta = 30*30
        last_mask = self.full_mask
        first = True
        
        while first or not (last_mask == self.full_mask).all():
            if first:
                first = False
            last_mask = self.full_mask
            
            # choose points within the full mask that is within the delta
            mask_idx = np.array(self.full_mask_robust.nonzero())
#             print(mask_idx)
            Np = self.p.shape[0]
            dist_Np = np.zeros((Np-1, mask_idx.shape[1]))
            for i in range(self.p.shape[0]-1):
                dist_Np[i] = dist_to_line_2d(mask_idx, self.p[i], self.p[i+1])
            dist = np.min(dist_Np, axis=0)
#             print(dist)
#             print(dist < delta)
            cons_mask_idx = mask_idx[:, dist < delta]
#             print(cons_mask_idx)
            cons_mask = np.zeros((self.H, self. W)).astype(bool)
            cons_mask[cons_mask_idx[0], cons_mask_idx[1]] = True
#             print('current consensus mask')
#             plt.imshow(cons_mask)
#             plt.show()
            diff_mask = np.logical_xor(cons_mask, ~self.full_mask_robust.astype(bool))
#             plt.imshow(diff_mask)
#             plt.show()
            self.set_obs(cons_mask, rgb_np = self.rgb_np, depth_np = self.depth_np, subsample=True)
            vis_img = self.draw_vis()
#             plt.imshow(vis_img)
#             plt.show()
            
            p = self.init_p.copy().reshape(-1)
            cons = []
            opt = {
                'maxiter': 5000,
                'disp': False,
                'ftol': 1.0
            }
            for i in range(p.shape[0]//2-1):
                fixed_len_i = {
                    'type': 'eq',
                    'fun': self.fixed_len_i,
                    'jac': self.fixed_len_i_jac,
                    'args': [i]
                }
                cons.append(fixed_len_i)
#             print(p)
            res = minimize(self.obj_fn, p, jac=self.jac_fn, constraints=cons, method='SLSQP', bounds=self.bound, options=opt)
#             print(res)
            self.p = res.x.reshape(-1, 2)
        
    def draw_vis(self, p=None):
        if self.rgb_np is not None:
            vis_img = self.rgb_np.copy()
        else:
            vis_img = np.zeros((self.H, self.W))
            
        radius = 5
        color = (255, 0, 0)
        thickness = 3
        
        if p is None:
            for i in range(self.p.shape[0]):
                vis_img = cv2.circle(vis_img, (int(self.p[i][1]), int(self.p[i][0])), radius, color, thickness)
        else:
            for i in range(p.shape[0]):
                vis_img = cv2.circle(vis_img, (int(p[i][1]), int(p[i][0])), radius, color, thickness)
        return vis_img
        
    def vis(self, save_dir=None, idx=0, p=None):
        vis_img = self.draw_vis(p)
        plt.imshow(vis_img)
        if save_dir is not None:
            import pathlib
            pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
            plt.savefig(save_dir+"frame_"+'{0:03d}'.format(idx)+".png", dpi=300)
        plt.show()
```

```python
# sanity check for LSLQP
# H = 540
W = 810
l = 20
simple_tracker = indi_pt_tracker(H, W, l, debug=True, reg=True, ratio=0.01, lam=1000.)
# q = np.array([201.95308515, 201.66400103, 202.09786527, 221.66352626,
#        202.41771163, 241.66245867, 201.92526732, 261.66401374,
#        202.13836598, 281.66290788, 202.59445072, 301.65770758,
#        202.72086699, 321.6640761 , 203.18105393, 341.68626015,
#        201.98515825, 361.66378504, 189.96367273, 377.6476552 ,
#        172.12468215, 386.72147204])
# q=q.reshape((11,2))
q = np.zeros((11,2))
q[:, 0] = 180
q[:, 1] = np.arange(180, 381, 20)
simple_tracker.set_init(q)
mask = np.zeros((H, W), dtype=bool)
mask[200:205, 200:340] = True
mask[210:215, 395:400] = True
simple_tracker.set_obs(mask, subsample=True)
# simple_tracker.trans_step()
simple_tracker.step(debug=False)
simple_tracker.vis()
```

```python
# simple rope tracking for LSLQP with reg
H = 540
W = 810
l = 20
simple_tracker_sub = indi_pt_tracker(H, W, l, reg=True, ratio=0.01, lam=1000.)
p = np.zeros((19,2))
p[:,0] = 10
p[:,1] = np.arange(40, 401, l)
simple_tracker_sub.set_init(p)
rand_h = random.randint(0, H-5)
rand_w = random.randint(0, W-5)
rand_h = 30
rand_w = 780

rand_h_np = np.random.randint(0, H-5, size=10).astype(int)
rand_w_np = np.random.randint(0, W-5, size=10).astype(int)

rand_h_coll = random.randint(0, H-50)
rand_w_coll = random.randint(0, W-50)
rand_h_coll = 350
rand_w_coll = 200

data_path = "/home/yixuan/dart_deformable/data/rope_simple/"
video_path = '/home/yixuan/dart_deformable/result/outliers/robust_rope_simple_collective_outlier.avi'
out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('M','J','P','G'), 20, (2*W, H))
for i in range(251):
    rgb_np = exr_to_np(data_path+"rgb_"+'{0:03d}'.format(i)+".exr")
    mask = rgb_np[:, :, 2] > 100
    
    if i > 0:
#         point outlier
#         mask[rand_h:rand_h+5, rand_w:rand_w+5] = True
    
#     contextual outlier
#     for i in range(rand_h_np.shape[0]):
#         mask[rand_h_np[i]:rand_h_np[i]+5, rand_w_np[i]:rand_w_np[i]+5] = True
    
#     collective outlier
        mask[rand_h_coll:rand_h_coll+100, rand_w_coll:rand_w_coll+100] = True
    
    simple_tracker_sub.set_obs(mask, rgb_np, subsample=True)
    start = time.time()
    if i > 0:
        simple_tracker_sub.robust_step(debug=False)
    else:
        simple_tracker_sub.step(debug=False)
    print("one iteration takes:", time.time()-start)
#     simple_tracker_sub.vis(save_dir="/home/yixuan/dart_deformable/result/reg_LSLQP_rope_comb_loss_19/", idx=i)
#     simple_tracker_sub.vis()
    vis_img = simple_tracker_sub.draw_vis()
    mask = np.repeat(mask[:, : ,None], 3, axis=2).astype(np.uint8)*255
    stack_img = np.hstack((mask, vis_img))
    out.write(stack_img)
#     plt.imshow(stack_img)
#     plt.show()
out.release()
```

```python
# simple occlusion for LSLQP with obs-based obj fn + reg
H = 540
W = 810
l = 20
simple_tracker_sub = indi_pt_tracker(H, W, l, debug=False,ratio=0.01, reg=True)
p = np.zeros((19,2))
p[:,0] = 275
p[:,1] = np.arange(40, 401, l)
simple_tracker_sub.set_init(p)
data_path = "/home/yixuan/blender_data/rope_simple_occlusion/render/"
for i in range(251):
    rgb_np = exr_to_np(data_path+"rgb_"+'{0:03d}'.format(i)+".exr")
    depth_np = depth_exr_to_np(data_path+"depth_"+'{0:03d}'.format(i)+".exr")
    hsv_np = cv2.cvtColor(rgb_np, cv2.COLOR_BGR2HSV)
    mask = np.bitwise_and(hsv_np[:, :, 1] > 150, hsv_np[:, :, 0] < 30)
    simple_tracker_sub.set_obs(mask, rgb_np, depth_np, subsample=True)
    start = time.time()
#     if i >= 25:
#         simple_tracker_sub.step(debug=True)
#     else:
    simple_tracker_sub.step(debug=False)
    print("one iteration takes:", time.time()-start)
#     simple_tracker_sub.vis(save_dir="/home/yixuan/dart_deformable/result/reg_LSLQP_rope_simple_occlusion_comb_loss/", idx=i)
    simple_tracker_sub.vis()
```

```python
# simple tracking in real data for LSLQP with model-based obj fn
H = 720
W = 1280
l = 40
track = indi_pt_tracker(H, W, l, ratio=0.01, reg=True)

start_idx = 414
# 1675 - 2022_01_05_14_35_00
end_idx = 842
# 2285 - 2022_01_05_14_35_00

q = np.zeros((19,2))
q[:,0] = 420
q[:,1] = np.arange(40, l*(q.shape[0]-1)+41, l)
track.set_init(q)
data_path = "/home/yixuan/Downloads/rope_dataset_0105_no_marker/2022_01_05_17_59_17/2022_01_05_17_59_17/"
video_path = '/home/yixuan/dart_deformable/result/LSLQP_rope_dataset_0105_no_occl_reg_comb_loss.avi'
out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('M','J','P','G'), 20, (2*W, H))
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
    
    track.set_obs(rope_mask, rgb_np, subsample=True)
    start = time.time()
#     if i == start_idx:
#         track.gauss_obj_step_trans()
#     if i >= 524 and i <= 542:
#         track.gauss_obj_step(debug = True)
#         print("one iteration takes:", time.time()-start)
#         track.vis(save_dir="/home/yixuan/dart_deformable/result/rope_dataset_0105_no_occl/", idx=i)
#     else:
#         track.gauss_obj_step()
#         print("one iteration takes:", time.time()-start)
#     if i > start_idx:
#         track.robust_step()
#     else:
    track.step()
    print("one iteration takes:", time.time()-start)
#     track.vis(save_dir="/home/yixuan/dart_deformable/result/LSLQP_rope_dataset_0105_no_occl_reg_comb_loss/", idx=i)
    vis_img=track.draw_vis()
    rope_mask = np.repeat(rope_mask[:, : ,None], 3, axis=2).astype(np.uint8)*255
    stack_img = np.hstack((rope_mask, vis_img))
    out.write(stack_img)
out.release()
```

```python
def imgs_to_video(path, start, end, img_ext='.png', img_prefix='frame'):
    rgb = cv2.imread(path+img_prefix+'_'+'{0:03d}'.format(start)+img_ext)
    H, W, _ = rgb.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(path+'dataset.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 20, (W, H))
    for i in range(start,end):
        rgb = cv2.imread(path+img_prefix+'_'+'{0:03d}'.format(i)+img_ext)
        out.write(rgb)
    out.release()
```

```python
path="/home/yixuan/dart_deformable/result/reg_LSLQP_rope_simple_occlusion_comb_loss/"
imgs_to_video(path, start=0, end=251+1, img_ext='.png', img_prefix='frame')
```

```python
# Albation study
# output: different computation time and accuracy under different r, lam; scale of total loss, three loss terms; correpsonding video

# ablation setting
# r_list = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
# lam_list = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 1e5]
r_list = [1.0]
lam_list = [1000]
out_dir = '/home/yixuan/dart_deformable/result/ablation/rope_simple_occlusion/'
in_dir = '/home/yixuan/blender_data/rope_simple_occlusion/render/'
pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

# tracker setting
H = 540
W = 810
l = 20
for r in r_list:
    for lam in lam_list:
        print('r:', r)
        print('lam:', lam)
        total_time = np.zeros(251)
        results = np.zeros((251, 19, 2))
        
        video_path = out_dir + 'r_' + str(r) + '_lam_' + str(lam) + '.avi'
        time_path = out_dir + 'time_r_' + str(r) + '_lam_' + str(lam) + '.npy'
        results_path = out_dir + 'results_r_' + str(r) + '_lam_' + str(lam) + '.npy'
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('M','J','P','G'), 20, (W, H))
        simple_tracker_sub = indi_pt_tracker(H, W, l, debug=False,ratio=r, reg=True, lam=lam)
        p = np.zeros((19,2))
        p[:,0] = 275
        p[:,1] = np.arange(40, 401, l)
        simple_tracker_sub.set_init(p)
        for i in range(251):
            rgb_np = exr_to_np(in_dir+"rgb_"+'{0:03d}'.format(i)+".exr")
            depth_np = depth_exr_to_np(in_dir +"depth_"+'{0:03d}'.format(i)+".exr")
            hsv_np = cv2.cvtColor(rgb_np, cv2.COLOR_BGR2HSV)
            mask = np.bitwise_and(hsv_np[:, :, 1] > 150, hsv_np[:, :, 0] < 30)
            simple_tracker_sub.set_obs(mask, rgb_np, depth_np, subsample=True)
            start = time.time()
        #     if i >= 25:
        #         simple_tracker_sub.step(debug=True)
        #     else:
            simple_tracker_sub.step(debug=False)
            total_time[i] = time.time()-start
            results[i] = simple_tracker_sub.p
            vis_img = simple_tracker_sub.draw_vis()
            
            out.write(vis_img)
        np.save(time_path, total_time)
        np.save(results_path, results)
        out.release()
            
#             print("one iteration takes:", time.time()-start)
        #     simple_tracker_sub.vis(save_dir="/home/yixuan/dart_deformable/result/reg_LSLQP_rope_simple_occlusion_comb_loss/", idx=i)
#             simple_tracker_sub.vis()
```

```python
# Albation study
# output: different computation time and accuracy under different r, lam; scale of total loss, three loss terms; correpsonding video

# ablation setting
# r_list = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
# lam_list = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 1e5]
r_list = [0.1, 1.0]
lam_list = [100, 1000, 10000, 1e5]
out_dir = '/home/yixuan/dart_deformable/result/ablation/rope_simple/'
in_dir = '/home/yixuan/blender_data/rope_simple/render/'
pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

# tracker setting
H = 540
W = 810
l = 20
for r in r_list:
    for lam in lam_list:
        print('r:', r)
        print('lam:', lam)
        total_time = np.zeros(251)
        results = np.zeros((251, 19, 2))
        
        video_path = out_dir + 'r_' + str(r) + '_lam_' + str(lam) + '.avi'
        time_path = out_dir + 'time_r_' + str(r) + '_lam_' + str(lam) + '.npy'
        results_path = out_dir + 'results_r_' + str(r) + '_lam_' + str(lam) + '.npy'
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('M','J','P','G'), 20, (W, H))
        simple_tracker_sub = indi_pt_tracker(H, W, l, debug=False,ratio=r, reg=True, lam=lam)
        p = np.zeros((19,2))
        p[:,0] = 275
        p[:,1] = np.arange(40, 401, l)
        simple_tracker_sub.set_init(p)
        for i in range(251):
            rgb_np = exr_to_np(in_dir+"rgb_"+'{0:03d}'.format(i)+".exr")
            mask = rgb_np[:, :, 2] > 100
            simple_tracker_sub.set_obs(mask, rgb_np, subsample=True)
            start = time.time()
        #     if i >= 25:
        #         simple_tracker_sub.step(debug=True)
        #     else:
            try:
                simple_tracker_sub.step(debug=False)
            except:
                break
            total_time[i] = time.time()-start
            results[i] = simple_tracker_sub.p
            vis_img = simple_tracker_sub.draw_vis()
            
            out.write(vis_img)
        np.save(time_path, total_time)
        np.save(results_path, results)
        out.release()
            
#             print("one iteration takes:", time.time()-start)
        #     simple_tracker_sub.vis(save_dir="/home/yixuan/dart_deformable/result/reg_LSLQP_rope_simple_occlusion_comb_loss/", idx=i)
#             simple_tracker_sub.vis()
```

```python
# ablation study result processing
# accuracy
out_dir = '/home/yixuan/dart_deformable/result/ablation/rope_simple_occlusion/'
data_path = '/home/yixuan/blender_data/rope_simple_occlusion/'
gt_path = data_path + 'ground_truth.npy'
cam_path = data_path + 'cam_info.txt'
gt = np.load(gt_path)
gt = gt + np.array([-0.5, 0.5, 0])[None, None, :]
gt = np.hstack((gt,np.zeros((251, 1, 3)))) # for boundary indexing of interpolation

# interpolate ground truth
scale = np.arange(19)*49/18
scale_int = np.floor(scale)
scale_diff = (scale-scale_int)[None, :, None]
gt_inter = gt[:, scale_int.astype(int), :] * (1-scale_diff) + gt[:, scale_int.astype(int)+1, :] * scale_diff

# convert to 2D
K = np.loadtxt(cam_path)
gt_inter_2d_homo = np.matmul(K[None, :, :], np.transpose(gt_inter, axes=(0,2,1)))
gt_inter_2d = np.zeros((251, 19, 2))
gt_inter_2d[:, :, 0] = gt_inter_2d_homo[:, 1, :]/gt_inter_2d_homo[:, 2, :]
gt_inter_2d[:, :, 1] = gt_inter_2d_homo[:, 0, :]/gt_inter_2d_homo[:, 2, :]

# read result
r_list = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
lam_list = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 1e5]

avg_time = np.zeros((10, 10))
avg_error = np.zeros((10, 10))

for r_idx, r in enumerate(r_list):
    for lam_idx, lam in enumerate(lam_list):
#         print('r:', r)
#         print('lam:', lam)
        time_path = out_dir + 'time_r_' + str(r) + '_lam_' + str(lam) + '.npy'
        results_path = out_dir + 'results_r_' + str(r) + '_lam_' + str(lam) + '.npy'
        try:
            results = np.load(results_path)
            times = np.load(time_path)
        except:
            print('NOT EXIST!')
            avg_time[r_idx, lam_idx] = float('NaN')
            avg_error[r_idx, lam_idx] = float('NaN')
            continue
        errors = np.linalg.norm(results-gt_inter_2d, axis=2).mean(axis=1)
#         print('avg time:', time.mean())
#         print('avg error:', errors.mean())
        avg_time[r_idx, lam_idx] = times.mean()
        avg_error[r_idx, lam_idx] = errors.mean()
        lbl_str = 'r_' + str(r) + '_lam_' + str(lam)
#         if r == 1e-2 and (lam == 1e2 or lam == 1e3 or lam == 1e4 or lam == 1e5):
#             plt.plot(errors, label = lbl_str)
        if lam == 1e3 and (r == 1e-3 or r == 1e-2 or r == 1e-1 or r ==1.0):
            plt.plot(errors, label = lbl_str)

plt.legend()
plt.savefig(out_dir+"comp_r.png", dpi=300)
plt.show()

idx=np.array([0, 3, 6, 9])
idx=idx.reshape((1,4)).repeat(4, axis=0)
idx_h = idx.T.reshape(-1)
idx_w = idx.reshape(-1)
print(avg_time[idx_h, idx_w].reshape((4,4)))
print(avg_error[idx_h, idx_w].reshape((4,4)))
```

```python

```
