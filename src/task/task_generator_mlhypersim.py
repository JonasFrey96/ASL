import copy
from task import TaskGenerator, Task

__all__ = ["TaskGeneratorMLHypersim"]

mlhypersim_template_dict = {
  "name": "scannet",
  "mode": "train",
  "output_size": [320, 640],
  "scenes": [],
  "data_augmentation": True,
  "label_setting": "default",
  "confidence_aux": 0,
}


class TaskGeneratorMLHypersim(TaskGenerator):
  def __init__(self, mode, cfg, *args, **kwargs):
    # SET ALL TEMPLATES CORRECT
    super(TaskGeneratorMLHypersim, self).__init__(cfg)

    for k in cfg["copy_to_template"].keys():
      mlhypersim_template_dict[k] = cfg["copy_to_template"][k]

    mode_cfg = cfg.get(mode, {})
    if mode == "mlhypersim_scenes":
      self._mylhypersim_scenes(**mode_cfg)
    else:
      raise AssertionError("TaskGeneratorMLHypersim: Undefined Mode")

    self.init_end_routine(cfg)

  def _mlhypersim_scenes(
    self, number_of_tasks, scenes_per_task, label_setting="default", confidence_aux=0
  ):
    train = copy.deepcopy(mlhypersim_template_dict)
    val = copy.deepcopy(mlhypersim_template_dict)
    train["mode"] = "train"
    val["mode"] = "val"
    train["label_setting"] = label_setting
    train["confidence_aux"] = confidence_aux

    val["label_setting"] = label_setting

    start_scene_train = 0
    for i in range(number_of_tasks):
      # GENERATE TRAIN TASK
      train["scenes"] = [
        f"scene{s:04d}"
        for s in range(start_scene_train, start_scene_train + scenes_per_task)
      ]
      val["scenes"] = train["scenes"]
      t = Task(
        name=f"Train_{i}",
        dataset_train_cfg=copy.deepcopy(train),
        dataset_val_cfg=copy.deepcopy(val),
      )
      self._task_list.append(t)
      start_scene_train += scenes_per_task


def test():
  import sys
  import os

  sys.path.insert(0, os.getcwd())
  sys.path.append(os.path.join(os.getcwd() + "/src"))

  mode = "mlhypersim_scenes"
  cfg = {
    "copy_to_template": {"output_size": [320, 640], "label_setting": "default"},
    "mlhypersim_scenes": {"number_of_tasks": 4, "scenes_per_task": 1},
  }
  tg = TaskGeneratorMLHypersim(mode=mode, cfg=cfg)

  print(tg)

  return True


if __name__ == "__main__":
  test()
