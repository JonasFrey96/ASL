from pathlib import Path
import os
from ucdr import UCDR_ROOT_DIR
from ucdr.kimera_semantics import ray_cast_scene

meshes = [
    str(s)
    for s in Path(os.path.join(UCDR_ROOT_DIR, "results/scene_data/pred_3_02_reprojected")).rglob("*_predict_mesh.ply")
]
meshes.sort()
maps = [m.replace("_predict_mesh.ply", "_serialized.data") for m in meshes]
label_generate_idtf = "pred_3_02_reprojected"
scenes = [s.split("/")[-1][:12] for s in maps]

gen_dir = f"{UCDR_ROOT_DIR}/results/labels_generated"

for mesh, ma, scene in zip(meshes, maps, scenes):
    print(mesh, ma, scene)
    ray_cast_scene(mesh, ma, scene, label_generate_idtf, gen_dir)
