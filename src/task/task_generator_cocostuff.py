import copy
from task import TaskGenerator, Task

__all__ = ["TaskGeneratorCocoStuff"]

cocostuff_template_dict = {
  "name": "cocostuff164k",
  "mode": "train",
  "output_size": [320, 640],
  "scenes": [],
  "data_augmentation": True,
}


class TaskGeneratorCocoStuff(TaskGenerator):
  def __init__(self, mode, cfg, *args, **kwargs):
    # SET ALL TEMPLATES CORRECT
    super(TaskGeneratorCocoStuff, self).__init__(cfg)

    mode_cfg = cfg.get(mode, {})
    if mode == "cocostuff_pretrain":
      self._coco_pretrain(**mode_cfg)

    else:
      raise AssertionError("TaskGeneratorCocoStuff: Undefined Mode")
    self._current_task = 0
    self._total_tasks = len(self._task_list)

    self.init_end_routine(cfg)

  def _coco_pretrain(self):
    train = copy.deepcopy(cocostuff_template_dict)
    val = copy.deepcopy(cocostuff_template_dict)
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
