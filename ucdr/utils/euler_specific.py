import os
from pytorch_lightning.utilities import rank_zero_only


@rank_zero_only
def move_datasets(env, dataset_list):
    if not env["workstation"]:
        tmp_dir = os.getenv("TMPDIR")
        scans = os.path.join(tmp_dir, "scannet", "scans")
        os.mkdir(scans_dir)
        os.mv(os.path.join(env["scannet_tar"], "scannetv2-labels.combined.tsv"), os.path.join(tmp_dir, "scannet"))

        for dataset in dataset_list:
            tar = os.path.join(env["scannet_tar"], dataset, ".tar")
            cmd = f"tar -xvf {tar} -C {scans} >/dev/null 2>&1"

        env["scannet"] = os.path.join(tmp_dir, "scannet")
