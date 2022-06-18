from ucdr.pseudo_label import label_generation
from ucdr.utils import load_yaml
from ucdr import UCDR_ROOT_DIR
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generate",
        default="pred1.yml",
        help="Experiment yaml file.",
    )
    args = parser.parse_args()
    generate_cfg_path = args.generate
    if not os.path.isabs(generate_cfg_path):
        generate_cfg_path = os.path.join(UCDR_ROOT_DIR, "cfg/generate", args.generate)

    gen = load_yaml(generate_cfg_path)

    lg = gen["label_generation"]

    label_generation(
        identifier=lg["identifier"],
        confidence=lg["confidence"],
        scenes=lg["scenes"],
        checkpoint_load=gen["checkpoint_load"],
        model_cfg=gen["model"]["cfg"],
    )
