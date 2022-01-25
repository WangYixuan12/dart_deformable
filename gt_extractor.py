import copy
import json
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import functools
import plotly
from open3d.web_visualizer import WebVisualizer
from open3d.visualization.async_event_loop import async_event_loop

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
                theta = -3*math.pi/4
                trans_init = np.identity(4)
                trans_init[0, 0] = math.cos(theta)
                trans_init[2, 0] = math.sin(theta)
                trans_init[0, 2] = -math.sin(theta)
                trans_init[2, 2] = math.cos(theta)
                trans_init[0, 3] = -1.0
                trans_init[2, 3] = 2.0

                radius = 0.01
                source_down = pc_o3d[-1].voxel_down_sample(radius)
                target_down = pc_o3d[0].voxel_down_sample(radius)

                source_down.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
                target_down.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
                source_down.remove_non_finite_points()
                target_down.remove_non_finite_points()
                reg_p2p = o3d.pipelines.registration.registration_colored_icp(
                    source_down, target_down, radius, trans_init,
                    o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                                      relative_rmse=1e-6,
                                                                      max_iteration=50))

#                 pc_src = copy.deepcopy(pc_o3d[-1])
                source_down.transform(reg_p2p.transformation)
                # source_down.transform(trans_init)
#                 visualizer = web_visualizer.WebVisualizer()
                o3d.visualization.draw_geometries([source_down, target_down])
#                 visualizer.show()

t_lists = sync_time()
reg_pc(t_lists)