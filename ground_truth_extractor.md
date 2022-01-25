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
import copy
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import functools
import plotly
from open3d.web_visualizer import WebVisualizer
from open3d.visualization.async_event_loop import async_event_loop
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
def gt_2d(path, start_idx, end_idx):
    '''
    path: directory path containing realsense_overhead_5_l515_color and realsense_overhead_5_l515_depth
    '''
    for i in range(start_idx, end_idx+1, 1):
        rgb_path = path + "realsense_overhead_5_l515_color/" + str(i) + ".jpg"
        depth_path = path + "realsense_overhead_5_l515_depth/" + str(i) + ".png"
        rgb = cv2.imread(rgb_path)
#         print(rgb.shape)
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        
#         plt.imshow(depth)
#         plt.colorbar()
#         plt.show()
        depth_mask = np.logical_and(depth < 1000, depth > 600)
        
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
#         plt.imshow(hsv[:, :, 0])
#         plt.colorbar()
#         plt.show()
#         plt.imshow(hsv[:, :, 1])
#         plt.colorbar()
#         plt.show()
#         plt.imshow(hsv[:, :, 2])
#         plt.colorbar()
#         plt.show()
        
        marker_min = np.array([20,0,200])
        marker_max = np.array([100,100,300])
        marker_mask = threshold(hsv, marker_min, marker_max)
        marker_mask = np.logical_and(marker_mask, depth_mask)
#         plt.imshow(marker_mask)
#         plt.show()
        
        rope_min = np.array([0,75,200])
        rope_max = np.array([60,250,300])
        rope_mask = threshold(hsv, rope_min, rope_max)
        rope_mask = np.logical_and(rope_mask, depth_mask)
#         rope_mask = np.logical_or(rope_mask, marker_mask)
#         plt.imshow(rope_mask)
#         plt.show()
```

```python
# # path = "/home/yixuan/Downloads/rope_dataset_0105/2022_01_05_14_35_00/2022_01_05_14_35_00/"
# # start_idx = 1675
# # end_idx = 2285

# path = "/home/yixuan/Downloads/rope_dataset_0105_no_marker/2022_01_05_17_59_17/2022_01_05_17_59_17/"
# start_idx = 414
# end_idx = 842

# gt_2d(path, start_idx, end_idx)
```

```python
def read_time(log_name):
    # read files
    f = open(log_name)
    lines = f.readlines()
    print("Reading robot log files...")
    # print(lines)
    times = []
    for line in lines:
        content = line.split('|')
        if 'robot' in content[-1]:
            times.append(float(content[-2]))
    return times
```

```python
def sync_time(data_dir = '/home/yixuan/Downloads/rope_dataset_0122/2022_01_22_11_13_56/2022_01_22_11_13_56/',
              cams=['realsense_overhead_5_l515', 'realsense_temp_left_yixuan', 'realsense_temp_right_yixuan']):
    # sync data source
    log_paths = []
    for cam in cams:
        log_paths.append(data_dir+cam+'_times.csv')
    times = []
    start_idx = 0
    end_idx = 100 # w.r.t first data source
    for log_path in log_paths:
        times.append(np.genfromtxt(log_path, delimiter='|')[1:,1])

    t1_idx_list = []
    t2_idx_list = []
    t3_idx_list = []

    t2_idx = 0
    t3_idx = 0

    d_t = 0.05
    for i in range(start_idx, end_idx):
        t1 = times[0][i]
        while t2_idx < len(times[1]) and times[1][t2_idx] < t1 and abs(times[1][t2_idx] - t1) > d_t:
            t2_idx += 1
        if abs(times[1][t2_idx] - t1 > d_t):
            continue
        while t3_idx < len(times[2]) and times[2][t3_idx] < t1 and abs(times[2][t3_idx] - t1) > d_t:
            t3_idx += 1
        if abs(times[2][t3_idx] - t1 > d_t):
            continue
        t1_idx_list.append(i)
        t2_idx_list.append(t2_idx)
        t3_idx_list.append(t3_idx)
    
    return (t1_idx_list, t2_idx_list, t3_idx_list)
```

```python
def draw(geometry=None,
         title="Open3D",
         width=640,
         height=480,
         actions=None,
         lookat=None,
         eye=None,
         up=None,
         field_of_view=60.0,
         bg_color=(1.0, 1.0, 1.0, 1.0),
         bg_image=None,
         show_ui=None,
         point_size=None,
         animation_time_step=1.0,
         animation_duration=None,
         rpc_interface=False,
         on_init=None,
         on_animation_frame=None,
         on_animation_tick=None):
    """Draw in Jupyter Cell"""

    window_uid = async_event_loop.run_sync(
        functools.partial(o3d.visualization.draw,
                          geometry=geometry,
                          title=title,
                          width=width,
                          height=height,
                          actions=actions,
                          lookat=lookat,
                          eye=eye,
                          up=up,
                          field_of_view=field_of_view,
                          bg_color=bg_color,
                          bg_image=bg_image,
                          show_ui=show_ui,
                          point_size=point_size,
                          animation_time_step=animation_time_step,
                          animation_duration=animation_duration,
                          rpc_interface=rpc_interface,
                          on_init=on_init,
                          on_animation_frame=on_animation_frame,
                          on_animation_tick=on_animation_tick,
                          non_blocking_and_return_uid=True))
    visualizer = WebVisualizer(window_uid=window_uid)
    visualizer.show()
```

```python
# register point cloud
def reg_pc(idx_lists, data_dir = '/home/yixuan/Downloads/rope_dataset_0122/2022_01_22_11_13_56/2022_01_22_11_13_56/',
           cams = ['realsense_overhead_5_l515', 'realsense_temp_left_yixuan', 'realsense_temp_right_yixuan'],
           cam_json_dir = '/home/yixuan/TRINA/Settings/sensors/'):
    
    # set up ICP registration
    threshold = 0.02
    
    rgb_paths = []
    depth_paths = []
    for cam in cams:
        rgb_paths.append(data_dir+cam+'_color/')
        depth_paths.append(data_dir+cam+'_depth/')
    
    # read intri
    intri_list = []
    for cam in cams:
        cam_json = cam_json_dir+cam+'.json'
        cam_json_file = open(cam_json)
        cam_config = json.load(cam_json_file)
        cam_json_file.close()
        intri_list.append(np.array(cam_config['720p']['mtx']))
    
#     for t_idx in range(len(idx_lists[0])):
    for t_idx in range(1):
        pc_o3d = []
        for src_idx in range(len(rgb_paths)):
            rgb = o3d.io.read_image(rgb_paths[src_idx]+str(t_idx)+'.jpg')
            depth = o3d.io.read_image(depth_paths[src_idx]+str(t_idx)+'.png')
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth)
            cam_intr = intri_list[src_idx]
            shape = rgb.get_max_bound()
            intr_o3d = o3d.camera.PinholeCameraIntrinsic(int(shape[0]), int(shape[1]), cam_intr[0,0], cam_intr[1,1], cam_intr[0,2], cam_intr[1,2])
            pc = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr_o3d)
            bbox = o3d.geometry.AxisAlignedBoundingBox(np.array([-10.0, -10.0, 0.0]), np.array([10.0, 10.0, 2.0]))
            pc = pc.crop(bbox)
            pc_o3d.append(pc)
#             pc_o3d[-1].point["colors"] = pc_o3d[-1].point["colors"].to(o3d.core.Dtype.Float32) / 255.0
            
            if src_idx > 0:
                trans_init = np.identity(4)
                radius = 0.02
                source_down = pc_o3d[-1].voxel_down_sample(radius)
                target_down = pc_o3d[0].voxel_down_sample(radius)

                source_down.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
                target_down.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
#                 reg_p2p = o3d.pipelines.registration.registration_icp(
#                     source_down, target_down, radius, trans_init,
#                     o3d.pipelines.registration.TransformationEstimationForColoredICP(),
#                     o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
#                                                                       relative_rmse=1e-6,
#                                                                       max_iteration=50))
#                     o3d.pipelines.registration.TransformationEstimationPointToPoint())
#                 pc_src = copy.deepcopy(pc_o3d[-1])
#                 source_down.transform(reg_p2p.transformation)
#                 visualizer = web_visualizer.WebVisualizer()
                draw([source_down, target_down])
#                 visualizer.show()

```

```python
t_lists = sync_time()
reg_pc(t_lists)
```

```python

```
