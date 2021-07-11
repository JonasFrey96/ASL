import copy
from task import TaskGenerator, Task
from datasets_asl import MLHypersim

__all__ = ["TaskGeneratorMLHypersim"]

mlhypersim_template_dict = {
  "name": "mlhypersim",
  "mode": "train",
  "output_size": [320, 640],
  "scenes": [],
  "data_augmentation": True,
}

classes = MLHypersim.get_classes()
rooms = list(set([c[:6] for c in classes]))
rooms.sort()


class TaskGeneratorMLHypersim(TaskGenerator):
  def __init__(self, mode, cfg, *args, **kwargs):
    # SET ALL TEMPLATES CORRECT
    super(TaskGeneratorMLHypersim, self).__init__(cfg)

    for k in cfg["copy_to_template"].keys():
      mlhypersim_template_dict[k] = cfg["copy_to_template"][k]

    mode_cfg = cfg.get(mode, {})
    if mode == "mlhypersim_scenes":
      self._mlhypersim_scenes(**mode_cfg)
    else:
      raise AssertionError("TaskGeneratorMLHypersim: Undefined Mode")

    self.init_end_routine(cfg)

  def _mlhypersim_scenes(self, number_of_tasks, scenes_per_task):
    train = copy.deepcopy(mlhypersim_template_dict)
    val = copy.deepcopy(mlhypersim_template_dict)
    train["mode"] = "train"
    val["mode"] = "val"

    start_scene_train = 0
    for i in range(number_of_tasks):
      # GENERATE TRAIN TASK
      sel_rooms = [
        rooms[r] for r in range(start_scene_train, start_scene_train + scenes_per_task)
      ]
      train["scenes"] = []
      for c in classes:
        for r in sel_rooms:
          if c.find(r) != -1:
            train["scenes"].append(c)

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
    "copy_to_template": {"output_size": [320, 640]},
    "mlhypersim_scenes": {"number_of_tasks": 4, "scenes_per_task": 1},
  }
  tg = TaskGeneratorMLHypersim(mode=mode, cfg=cfg)

  print(tg)

  return True


if __name__ == "__main__":
  test()
