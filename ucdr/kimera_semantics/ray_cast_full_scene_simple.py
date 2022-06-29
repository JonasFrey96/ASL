import numpy as np
import os
from pathlib import Path
from multiprocessing import Pool
import imageio
import time
import argparse

from ucdr.kimera_semantics.raycast_scene import LabelGenerator
from ucdr.kimera_semantics import store_hard_label
from ucdr import UCDR_ROOT_DIR
from ucdr.utils import load_env


def ray_cast_scene(mesh_path, map_serialized_path, scene, label_generate_idtf, label_generate_dir, r_sub=5):

    label_generator = LabelGenerator(mesh_path, map_serialized_path, scene, r_sub, visu3d=False)
    out_dir = os.path.join(
        label_generate_dir,
        label_generate_idtf,
        "scans",
        scene,
        label_generate_idtf,
    )
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    scannet_scene_dir = os.path.join(load_env()["scannet"], "scans", scene)
    poses = [
        str(p) for p in Path(f"{scannet_scene_dir}/pose/").rglob("*.txt") if int(str(p).split("/")[-1][:-4]) % 10 == 0
    ]
    poses.sort(key=lambda p: int(p.split("/")[-1][:-4]))
    nr = len(poses)

    for j, p in enumerate(poses):
        H_cam = np.loadtxt(p)
        st = time.time()
        probs = label_generator.get_label(H_cam)
        gen_time = time.time() - st
        st = time.time()
        p_out = os.path.join(out_dir, p.split("/")[-1][:-4] + ".png")
        store_hard_label(probs[:, :, :].argmax(axis=2), p_out)
        print(f"{j}/{nr}", " Store time", time.time() - st, " Gen time: ", gen_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # EXTERNAL DATA PATHS
    parser.add_argument(
        "--mesh_path",
        type=str,
        default=f"{UCDR_ROOT_DIR}/results/scene_data/pred_2_02_reprojected/scene0004_00_pred_2_02_reprojected_predict_mesh.ply",
        help="",
    )
    parser.add_argument(
        "--map_serialized_path",
        type=str,
        default=f"{UCDR_ROOT_DIR}/results/scene_data/pred_2_02_reprojected/scene0004_00_pred_2_02_reprojected_serialized.data",
        help="",
    )
    parser.add_argument("--label_generate_idtf", type=str, default="scene0004_00_pred_2_02_reprojected", help="")
    parser.add_argument("--label_generate_dir", type=str, default=f"{UCDR_ROOT_DIR}/results/labels_generated", help="")
    parser.add_argument("--scene", type=str, default="scene0004_00", help="")
    parser.add_argument("--r_sub", type=int, default=5, help="")

    args = parser.parse_args()

    ray_cast_scene(
        args.mesh_path,
        args.map_serialized_path,
        args.scene,
        args.label_generate_idtf,
        args.label_generate_dir,
        args.r_sub,
    )
