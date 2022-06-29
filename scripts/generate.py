from ucdr.pseudo_label import label_generation
from ucdr.utils import load_yaml
from ucdr import UCDR_ROOT_DIR
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generate",
        default="pred1.yaml",
        help="Experiment yaml file.",
    )
    args = parser.parse_args()
    generate_cfg_path = args.generate
    if not os.path.isabs(generate_cfg_path):
        generate_cfg_path = os.path.join(UCDR_ROOT_DIR, "cfg/generate", args.generate)

    gen_cfg = load_yaml(generate_cfg_path)

    global_checkpoint = gen_cfg.get("global_checkpoint_load", None)
    for gen in gen_cfg["label_generations"]:
        print(gen["identifier"], gen["scenes"])

        checkpoint_load = gen.get("checkpoint_load", global_checkpoint)
        if not os.path.isabs(checkpoint_load):
            checkpoint_load = os.path.join(UCDR_ROOT_DIR, checkpoint_load)

        label_generation(
            identifier=gen["identifier"],
            scenes=gen["scenes"],
            checkpoint_load=checkpoint_load,
            model_cfg=gen_cfg["model"]["cfg"],
        )
