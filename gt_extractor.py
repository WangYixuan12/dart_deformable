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
from open3d.visualization import Visualizer
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
    Nc = len(cams)
    # sync data source
    log_paths = []
    for cam in cams:
        log_paths.append(data_dir+cam+'_times.csv')
    times = []
    start_idx = 150
    end_idx = 250 # w.r.t first data source
    for log_path in log_paths:
        times.append(np.genfromtxt(log_path, delimiter='|')[1:,1])

    t_lists = []
    t_idx = []
    for i in range(Nc):
        t_lists.append([])
        if i > 0:
            t_idx.append(0)

    d_t = 0.05
    for i in range(start_idx, end_idx):
        t1 = times[0][i]
        synced = True
        for c in range(Nc-1):
            while t_idx[c] < len(times[1]) and times[1][t_idx[c]] < t1 and abs(times[1][t_idx[c]] - t1) > d_t:
                t_idx[c] += 1
            if abs(times[1][t_idx[c]] - t1 > d_t):
                synced = False
                break
        
        if synced:
            for c in range(Nc):
                if c == 0:
                    t_lists[c].append(i)
                else:
                    t_lists[c].append(t_idx[c-1])
    
    return t_lists

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
    
    curr_trans = []
    # trans_init_1 = np.array([[-0.70316719, -0.20443537,  0.68100079, -0.93927501],
    #             [ 0.19127522, 0.86807883, 0.45809708, -0.42976493],
    #             [-0.68481362, 0.45237741,-0.57130113, 1.77694394],
    #             [ 0.        , 0.        , 0.        , 1.        ]])

    trans_init_1 = np.array([[-0.64835677, -0.21537223,  0.73023852, -1.18378519],
                            [ 0.22571122,  0.86165716,  0.45453425, -0.36487501],
                            [-0.72710931,  0.45952338, -0.51004932,  1.52784377],
                            [ 0.,          0.,          0.,          1.        ]])
    curr_trans.append(trans_init_1)
    curr_trans.append(trans_init_1)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for t_idx in range(len(idx_lists[0])):
        # for t_idx in range(1):
        pc_o3d = []
        for src_idx in range(len(rgb_paths)):
            rgb = o3d.io.read_image(rgb_paths[src_idx]+str(idx_lists[src_idx][t_idx])+'.jpg')
            depth = o3d.io.read_image(depth_paths[src_idx]+str(idx_lists[src_idx][t_idx])+'.png')
            if rgb.is_empty() or depth.is_empty():
                break
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth, convert_rgb_to_intensity=False)
            cam_intr = intri_list[src_idx]
            shape = rgb.get_max_bound()
            intr_o3d = o3d.camera.PinholeCameraIntrinsic(int(shape[0]), int(shape[1]), cam_intr[0,0], cam_intr[1,1], cam_intr[0,2], cam_intr[1,2])
            pc = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr_o3d)
            # pc.points["colors"] = pc.points["colors"].to(o3d.core.Dtype.Float32) / 255.0
            bbox = o3d.geometry.AxisAlignedBoundingBox(np.array([-10.0, -10.0, 0.0]), np.array([10.0, 10.0, 2.0]))
            pc = pc.crop(bbox)
            pc_o3d.append(pc)
            voxel_radius = [0.04, 0.02, 0.01]
            max_iter = [500, 300, 100]
            
            if src_idx > 0:
                # theta = -3*math.pi/4
                # trans_init = np.identity(4)
                # trans_init[0, 0] = math.cos(theta)
                # trans_init[2, 0] = math.sin(theta)
                # trans_init[0, 2] = -math.sin(theta)
                # trans_init[2, 2] = math.cos(theta)
                # trans_init[0, 3] = -1.0
                # trans_init[2, 3] = 2.0
                trans_init = curr_trans[src_idx-1]
                for i in range(len(max_iter)):
                    print('scale:', i)
                    radius = voxel_radius[i]
                    iter = max_iter[i]
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
                        o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-2,
                                                                        relative_rmse=1e-2,
                                                                        max_iteration=iter))
                    trans_init = reg_p2p.transformation
                    # reg_p2p = o3d.pipelines.registration.registration_icp(
                    #         source_down, target_down, threshold, trans_init,
                    #         o3d.pipelines.registration.TransformationEstimationPointToPoint())

                src = copy.deepcopy(pc_o3d[-1])
                tgt = copy.deepcopy(pc_o3d[0])
                src.transform(trans_init)
                print(trans_init)
                vis.clear_geometries()
                vis.add_geometry(src)
                vis.add_geometry(tgt)
                view_pt = vis.get_view_control() # o3d.visualization.ViewControl()
                view_pt.set_front(np.array([ 0.12788955966509832, -0.69995967661305747, -0.70263839323254418 ]))
                view_pt.set_lookat(np.array([ -0.073481474217788056, 0.13849170569234565, 0.7504930014607416 ]))
                view_pt.set_up(np.array([ -0.0060979615841821599, -0.70899749921605271, 0.70518462899435153 ]))
                view_pt.set_zoom(0.37999999999999967)
                vis.poll_events()
                vis.update_renderer()
                o3d.visualization.draw_geometries([src, tgt],
                                                front = [ 0.12788955966509832, -0.69995967661305747, -0.70263839323254418 ],
                                                lookat = [ -0.073481474217788056, 0.13849170569234565, 0.7504930014607416 ],
                                                up = [ -0.0060979615841821599, -0.70899749921605271, 0.70518462899435153 ],
                                                zoom = 0.37999999999999967)
                # curr_trans[src_idx-1] = trans_init
    vis.close()

# data_dir = '/home/yixuan/Downloads/rope_dataset_0122/2022_01_22_11_07_36/2022_01_22_11_07_36/'
data_dir = '/home/yixuan/Downloads/rope_dataset_0122/2022_01_22_11_13_56/2022_01_22_11_13_56/'
cams=['realsense_overhead_5_l515', 'realsense_temp_left_yixuan']

t_lists = sync_time(data_dir = data_dir,
                    cams = cams)
reg_pc(t_lists, data_dir = data_dir, cams = cams)