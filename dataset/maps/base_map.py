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

    def __init__(self, embeds_dir: str = "data/v2/maps"):

        self.results_dir = embeds_dir             
        self.model_name = "BLIP2"
        self.scene_name = None
        self.init_dict, self.embed_dict = None, None
        self.grid_size = None
        self.tf_hab_to_agent = None

    def load_embed_init(self,
                        scene_name: str,
                        episode_id: str = None,
                        base_dir: str = None):
        r"Loads the model (if not loaded), transformation matrix, embedding and init dicts"
        
        self.scene_name = scene_name     
        self.episode_id = episode_id
        self.init_dict = self.load_dict('init', base_dir)
        self.embed_dict = self.load_dict('embed', base_dir)

        print(f"Loaded Embed and Init dicts for Scene: {self.scene_name} and Episode: {self.episode_id}")

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
        r"Loads embed or init dict from the embeds_dir corresponding to the current scene and model name"
        
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
        r"Transforms Habitat coordinates to Agent coordinates"

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
        print(f'Loading array of shape: {embed_arr.shape}')
        
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

    def visualize(self, save_dir = None, plot_pts = None, 
                  plot_grid: bool = True, plot_reorient: bool = False,
                  point_size: int = 10):

        assert len(self.sim_dict) > 0, 'Please load the sim dict first'


        fig, ax = plt.subplots(figsize = (8, 8))

        top_pts = self.top_k_sims(k = 4)
        if plot_reorient: top_pts = self.rot_pts(top_pts)

        ax.scatter(top_pts[:, 0], top_pts[:, 1], 
                        facecolors='white', edgecolors='black', 
                        s=150, linewidths=2.5)
            
        self.plot_scores_in_grid(ax = ax, fig = fig, 
                                    plot_grid = plot_grid,
                                    plot_reorient=plot_reorient,
                                    point_size=point_size)

        if plot_pts is not None:
            if plot_reorient: plot_pts = self.rot_pts(plot_pts)
            ax.scatter(plot_pts[:, 0], plot_pts[:, 1], 
                        facecolors='white', edgecolors='green', 
                        s=150)
            ax.scatter(plot_pts[:, 0], plot_pts[:, 1], color = 'black')



        text_add = f'Prompt Mode: Multi_Modal\nText Prompt: {self.text}' if self.prompt_mode == "multi" \
                    else f'Prompt Mode: Text\nText Prompt: {self.text}' if self.prompt_mode == "text" \
                    else f'Prompt Mode: Image'
        
        fig.suptitle(f'Scene: {self.scene_name}, Model: {self.model_name}, {text_add}\n', 
                     fontstyle='italic', fontsize = 15)
        fig.tight_layout()

        if save_dir is not None:
            if not save_dir.endswith('png'):
                save_dir = os.path.join(save_dir, f'scene_{self.scene_name}_model_{self.model_name}.png')
            fig.savefig(save_dir)

            plt.close(fig)
        
        else: plt.show()

    def plot_img(self, ax: plt.Axes, title: str = 'Image Prompt'):
        ax.imshow(self.img)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
        
    def plot_scores_in_grid(self, ax: plt.Axes, fig: plt.figure, 
                            plot_grid: bool = False, plot_reorient: bool = False,
                            point_size: int = 10):

        #Get grid points and corresponding similarity scores
        grid_pts = np.array(list(self.sim_dict[0].keys()))
        sim_scores = np.array(list(self.sim_dict[0].values()))

        if plot_reorient:

            plot_grid = False
            
            # quat = self.init_dict['init_pose'].item()['init_rot_abs']
            # quat = [quat[0], quat[2], quat[1], quat[3]]

            # rot_vec = R.from_quat(quat).as_rotvec(degrees = False)
            # rot_vec[2] = rot_vec[2] + (np.pi/2)
            # rot_mat = R.from_rotvec(rot_vec).as_matrix()[:2, :2]

            # grid_pts = grid_pts @ rot_mat

            grid_pts = self.rot_pts(grid_pts)
        
        score_min, score_max = sim_scores.min(), sim_scores.max()
        score_lims = (score_min, score_max)


        x_min, y_min = grid_pts.min(axis = 0) - self.grid_size
        x_max, y_max = grid_pts.max(axis = 0) + 2 * self.grid_size
        x_lims = (x_min, x_max)
        y_lims = (y_min, y_max)

        #Plotting Grid with Similarity Scores
        if plot_grid:

            eu.plot_grid_lims(ax, 
                        x_lims = x_lims,
                        y_lims = y_lims,
                        grid_size = self.grid_size,
                        plot_all_centers = False)
        
        else:

            ax.set_xlim(x_lims[0], x_lims[1])
            ax.set_ylim(y_lims[0], y_lims[1])
        
        # Define a colormap
        cmap = cm.coolwarm_r
        norm = colors.Normalize(vmin = score_lims[0], vmax = score_lims[1])
        
        score_colors = cmap(norm(sim_scores))
        score_colors = np.array(score_colors).squeeze(axis = 1)
        
        # Add colorbar (legend for colormap)
        cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, orientation = "vertical")
        cbar.set_label("Embedding Similarity", labelpad=10)  # Label for the color legend
        cbar.ax.yaxis.set_label_position('right')

        # cbar.ax.set_yticks([val for val in np.linspace(score_lims[0], score_lims[1], 6)])
        # cbar.ax.set_yticklabels([np.round(val, 2) for val in np.linspace(score_lims[0], score_lims[1], 5)])

        cbar.set_ticks([val for val in np.linspace(score_lims[0], score_lims[1], 5)])
        cbar.set_ticklabels([np.round(val, 2) for val in np.linspace(score_lims[0], score_lims[1], 5)])

        #Plot grid point and corresponding color
        ax.scatter(grid_pts[:, 0], grid_pts[:, 1], color = score_colors, s = point_size)
        
        ax.set_title(f'Embedding Similarity Grid')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks([])
        ax.set_yticks([])

    def rot_pts(self, pts):

        quat = self.init_dict['init_pose'].item()['init_rot_abs']
        quat = [quat[0], quat[2], quat[1], quat[3]]

        rot_vec = R.from_quat(quat).as_rotvec(degrees = False)
        #rot_vec[2] = rot_vec[2] + (np.pi/2)
        rot_mat = R.from_rotvec(rot_vec).as_matrix()[:2, :2]

        pts = pts @ rot_mat

        return pts

