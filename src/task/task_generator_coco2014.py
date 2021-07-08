import copy
from task import TaskGenerator, Task

__all__ = ["TaskGeneratorCoco2014"]

coco_template_dict = {
  "name": "coco2014",
  "mode": "train",
  "output_size": [320, 640],
  "scenes": [],
  "data_augmentation": True,
}


class TaskGeneratorCoco2014(TaskGenerator):
  def __init__(self, mode, cfg, *args, **kwargs):
    # SET ALL TEMPLATES CORRECT
    super(TaskGeneratorCoco2014, self).__init__(cfg)

    mode_cfg = cfg.get(mode, {})
    if mode == "coco2014_pretrain":
      self._coco_pretrain(**mode_cfg)

    else:
      raise AssertionError("TaskGeneratorCoco2014: Undefined Mode")

    self.init_end_routine(cfg)

  def _coco_pretrain(self):
    train = copy.deepcopy(coco_template_dict)
    val = copy.deepcopy(coco_template_dict)
    train["mode"] = "train"
    val["mode"] = "val"
    i = 0
    t = Task(
      name=f"Train_{i}",
      dataset_train_cfg=copy.deepcopy(train),
      dataset_val_cfg=copy.deepcopy(val),
    )
    self._task_list.append(t)


def test():
  import sys
  import os

  sys.path.insert(0, os.getcwd())
  sys.path.append(os.path.join(os.getcwd() + "/src"))

  from utils_asl import load_yaml

  exp = load_yaml(os.path.join(os.getcwd() + "/cfg/test/test.yml"))
  env = load_yaml(os.path.join("cfg/env", os.environ["ENV_WORKSTATION_NAME"] + ".yml"))

  # TODO: Jonas Frey
  return True


if __name__ == "__main__":
  test()
