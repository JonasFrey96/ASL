import os
import argparse
from pathlib import Path
from ucdr import UCDR_ROOT_DIR
from ucdr.utils import load_yaml


if __name__ == "__main__":

    folder = "/home/jonfrey/git/ASL/cfg/exp/pred_3_r02"
    paths = [str(s) for s in Path(folder).rglob("*.yaml")]
    paths.sort()
    send = True
    sync = True
    if sync:
        os.system(os.path.join(UCDR_ROOT_DIR, "cfg/docker/sync_cluster.sh"))

    for p in paths[::-1]:
        exp = load_yaml(p)
        name = exp["name"]
        cmd = f"ssh euler mkdir -p /cluster/work/rsl/jonfrey/ucdr/learning/{name} "

        if send:
            os.system(cmd)
        print(cmd)

        outfile = f"/cluster/work/rsl/jonfrey/ucdr/learning/{name}/log.out"
        time = "4:00"

        file = p[p.find("cfg/exp") + 8 :]

        cmd = f"export OUTFILE_NAME={outfile} && export TIME={time} && {UCDR_ROOT_DIR}/cfg/docker/bsub.sh --exp={file}"

        if send:
            os.system(cmd)
        print(cmd)
