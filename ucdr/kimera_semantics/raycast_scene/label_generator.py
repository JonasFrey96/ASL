from ucdr.kimera_semantics.raycast_scene import VoxelMap
from ucdr.kimera_semantics.raycast_scene import RayCaster
from ucdr.kimera_semantics.raycast_scene import Visualizer3D
from ucdr.utils import load_env

import numpy as np
import imageio
import os


class LabelGenerator:
    def __init__(self, mesh_path, map_serialized_path, scene, r_sub, visu3d=True):
        # Pass the args config

        env = load_env()
        scannet_scene_dir = os.path.join(env["scannet"], "scans", scene)
        # GT LABEL TEST
        label_gt = imageio.imread(os.path.join(scannet_scene_dir, "label-filt/0" + ".png"))
        H, W = label_gt.shape
        size = (H, W)

        data = np.loadtxt(f"{ scannet_scene_dir }/intrinsic/intrinsic_depth.txt")
        k_render = np.array([[data[0, 0], 0, data[0, 2]], [0, data[1, 1], data[1, 2]], [0, 0, 1]])
        data = np.loadtxt(f"{scannet_scene_dir}/intrinsic/intrinsic_color.txt")
        k_image = np.array([[data[0, 0], 0, data[0, 2]], [0, data[1, 1], data[1, 2]], [0, 0, 1]])

        self.r_sub = r_sub
        self._voxel_map = VoxelMap(map_serialized_path, size, r_sub)
        self._ray_caster = RayCaster(mesh_path, k_image, size, r_sub)

        self._visu_active = visu3d
        if visu3d == True:
            self._visu3d = Visualizer3D(size, k_image, mesh_path)

    def get_label(self, H_cam):
        locations, index_ray, index_tri, ray_origins = self._ray_caster.raycast(H_cam)

        probs = self._voxel_map.ray_cast_results_to_probs(locations, index_ray)
        if self._visu_active:
            self._visu3d.visu(locations, ray_origins)

        return probs


if __name__ == "__main__":
    import argparse
    from ucdr.utils import load_env

    parser = argparse.ArgumentParser()
    # EXTERNAL DATA PATHS
    parser.add_argument(
        "--mesh_path",
        type=str,
        default="/home/jonfrey/Datasets/output_kimera_semantics/scene0000_00_pcmr_fixed_epochs_rerun_all_predict_mesh.ply",
        help="",
    )
    parser.add_argument(
        "--map_serialized_path",
        type=str,
        default="/home/jonfrey/Datasets/output_kimera_semantics/scene0000_00_pcmr_fixed_epochs_rerun_all_serialized.data",
        help="",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="scene0000_00",
        help="",
    )
    args = parser.parse_args()

    label_generator = LabelGenerator(args.mesh_path, args.map_serialized_path, args.scene)
    i = 10
    env = load_env()
    scannet_dir = env["scannet"]
    H_cam = np.loadtxt(f"{scannet_dir}/scans/scene0000_00/pose/{i}.txt")
    probs = label_generator.get_label(H_cam)
    print("Done")
