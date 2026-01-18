# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

# Library imports
from typing import Any, List
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
from matplotlib import cm, colors
import matplotlib.pyplot as plt

# Local imports
import dataset.maps.utils.embed_utils as eu
import dataset.maps.utils.geometry_utils as gu

class HabtoGrid:

    def __init__(self, embeds_dir: str = "data/val/maps"):

        self.results_dir = embeds_dir             
        self.model_name = "BLIP2"
        self.scene_name = None
        self.init_dict, self.embed_dict = None, None
        self.grid_size = None
        self.tf_hab_to_agent = None

    def load_embed_init(self,
                        scene_name: str,
                        episode_id: str = None,
                        base_dir: str = None
                        ):
        
        self.scene_name = scene_name     
        self.episode_id = episode_id
        self.init_dict = self.load_dict('init', base_dir)
        self.embed_dict = self.load_dict('embed', base_dir)

        #Tranformation matrix from habitat to agent coordinates
        hab_init_pose = self.init_dict['init_pose'].item()
        self.tf_hab_to_agent = self.load_tf_hab_to_agent(hab_init_pose)

        try:
            self.grid_size = int(self.init_dict['grid_cell_size'])
        except KeyError:
            self.grid_size = int(self.init_dict['pixels_per_meter'])
            print(f" Grid Size not found in init dict keys. \n Assuming grid size to be 1 m.sq : {self.grid_size} pixels")
            self.init_dict["grid_cell_size"] = self.init_dict["pixels_per_meter"]

    def load_dict(self,
                  dict_type: str,
                  base_dir: str = None,
                  ):
        
        assert dict_type in ['embed', 'init']
        assert self.model_name in ['BLIP2']

        dict_dir = os.path.join(self.results_dir, f'{dict_type}_dicts')

        if dict_type == "embed": 
            dict_dir = os.path.join(dict_dir, f'{self.scene_name}')
            file_name = f'{dict_type}_dict_model_{self.model_name}_scene_{self.scene_name}_episode_{self.episode_id}.npz'

            valid_files = [f for f in os.listdir(dict_dir) if f.__contains__(file_name)]
            assert len(valid_files) == 1, f"More than 1 file found for {self.scene_name} and {self.episode_id}"
            dict_path = os.path.join(dict_dir, file_name)
        else:
            file_name = f'{dict_type}_dict_model_{self.model_name}_scene_{self.scene_name}_episode_{self.episode_id}.npz'
            dict_path = os.path.join(dict_dir, file_name)

        dict_file = np.load(dict_path, allow_pickle=True)
        if dict_type == 'embed':
            return {eval(k) : v.squeeze(0) for k, v in dict_file.items()}
        
        return dict(dict_file)
        
    def load_tf_hab_to_agent(self, init_pose):

        #Get x, y coords
        x, y = np.array(init_pose['init_pos_abs'])[[0, 2]]
        robot_xy = np.array([-x, -y, 0])

        #Get yaw
        quat = init_pose['init_rot_abs']
        quat = [quat[0], quat[2], quat[1], quat[3]]
        rot = R.from_quat(quat).as_rotvec(degrees = False)
        robot_yaw = rot[2]
        
        #First translate, then rotate
        tf_mat_trans = gu.xyz_yaw_to_tf_matrix(xyz = robot_xy, yaw = 0)
        tf_mat_rot = gu.xyz_yaw_to_tf_matrix(xyz = np.zeros_like(robot_xy), yaw = robot_yaw)

        tf_mat = tf_mat_rot @ tf_mat_trans

        return tf_mat

    def px_to_hab(self, px_pts):
        r"""
        Converts from pixels to habitat coordinates
        Pipeline:
            -> Swap pixel coords (accounting for using cv2 contours)
            -> Convert pixels to agent coordinates
                -> Inverts y axis 
                -> Recenters to pixel origin 
                -> Rescales pixels to meters
                -> Swaps the x and y positions: (y, x) -> (x, y)
            -> Convert agent coords to habitat coords
        """

        px = px_pts.copy()
        px = px[:, ::-1]

        #Pixels to Agent Coords
        px[:, 0] = self.init_dict['map_shape'][0] - px[:, 0]
        
        xy_pts = (px - self.init_dict['pixel_origin']) / self.init_dict['pixels_per_meter']
        xy_pts = xy_pts[:, ::-1]

        #Agent to Hab Coords
        tf_mat_inv = np.linalg.inv(self.tf_hab_to_agent)
        xy_pts = np.hstack((xy_pts, np.zeros((xy_pts.shape[0], 1))))
        hab_pts = list(map(lambda pt: gu.transform_points(tf_mat_inv, pt[np.newaxis, :]), xy_pts))
        hab_pts = np.array(hab_pts).squeeze(1)[:, :2]

        return hab_pts
    
    def hab_to_px(self, hab_pts):
        r"""
        Converts from habitat coords to pixel coordinates
        Pipeline:
            -> Convert habitat to agent coordinates
            -> Convert agent to pixel coordinates
                -> Swap coords
                -> Rescales to pixel scale, and recenters to pixel origin
                -> Inverts the y axis
            -> Swap pixel coords
        """

        #Habitat to Agent coords
        hab = np.hstack((hab_pts, np.zeros((hab_pts.shape[0], 1))))
        xy = list(map(lambda pt: gu.transform_points(self.tf_hab_to_agent, pt[np.newaxis, :]), hab))
        xy = np.array(xy).squeeze(1)[:, :2]
        
        #Agent to Pixel coords
        xy = xy[:, ::-1]
        px = np.rint(xy * self.init_dict['pixels_per_meter']) + self.init_dict['pixel_origin']
        px[:, 0] = self.init_dict['map_shape'][0] - px[:, 0]

        return px.astype(int)[:, ::-1]
    
    def px_to_arr(self, px_pts, arr_origin):
        r"""
        Returns pixels values accounting for the grid size and array origin
        """

        px = px_pts.copy()
        px = px[:, ::-1]

        px[:, 0] = self.init_dict['map_shape'][0] - px[:, 0]
        px_rel = (px - self.init_dict['pixel_origin']) / self.init_dict['pixels_per_meter']
        px_rel = px_rel[:, ::-1]

        arr_pts = px_rel + arr_origin

        return arr_pts
    
    def load_embed_np_arr(self, visualize=False):
        r"""
        Creates a numpy array with embeddings at relevant positions (zeros at other positions).
        This is supposed to represent the map accounting for the grid size.
        E.g.
            -> Map of size 1000x1000
            -> Grid size of 20x20
            -> Embedding dimension of 768
            Creates an array of shape (50, 50, 768)
        """
        
        arr_shape = self.init_dict['map_shape'] // self.grid_size
        arr_origin = arr_shape // 2
        embed_dim = next(iter(self.embed_dict.values())).shape[0]

        embed_arr = np.zeros((arr_shape[0], arr_shape[1], embed_dim))
        
        px_to_arr_pos = {px : self.px_to_arr(np.array([px]), arr_origin)[0] \
                         for px in self.embed_dict.keys()}
        
        for px_pos in self.embed_dict.keys():
            arr_pos = px_to_arr_pos[px_pos]
            embed_arr[int(arr_pos[0]), int(arr_pos[1])] = self.embed_dict[px_pos]

        if visualize:
            vis_arr = embed_arr.sum(2)
            vis_arr[vis_arr > 0] = 1
            plt.imshow(vis_arr)
        
        return embed_arr
    
    def visualize(self, arr=None, target=None, save_to_disk=False, path_to_image="trainer/visualizations/map.png"):
        """
        Visualizes the numpy array as an image.
        If save_to_disk is True, saves the image to the specified path.
        """
        assert arr is not None, "Array to visualize cannot be None"
        
        # Load the map
        map_img = arr.sum(2)
        map_img[map_img > 0] = 1
        
        # Add the target position
        if target is not None:
            map_img[int(target[0]), int(target[1])] = 0.5
        
        # Save the image
        if save_to_disk:
            plt.imsave(path_to_image, map_img, cmap='jet', vmin=0, vmax=1)
            
        
            
            
        
        
