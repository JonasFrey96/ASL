""" 
Defines Data Strucutre for Tasks.
Task dosen`t know anything about CL!
Each Task should return a cfg such that a dataset can be initalized based on it. 
"""

__all__ = ["Task", "TaskGenerator"]


class Task:
  "Might also be implemeted as a dataclass."

  def __init__(self, name, dataset_train_cfg, dataset_val_cfg):
    self.name = name
    self.dataset_train_cfg = dataset_train_cfg
    self.dataset_val_cfg = dataset_val_cfg


class TaskGenerator:
  def __init__(self, cfg):
    self._task_list = []
    self._current_task = 0
    self._total_tasks = 0

  def init_end_routine(self, cfg):
    self._current_task = 0
    self._total_tasks = len(self._task_list)

  def __iter__(self):
    return self

  def __next__(self):
    if self._current_task < len(self._task_list):
      task = self._task_list[self._current_task]
      self._current_task += 1
      return task

    else:
      raise StopIteration

  def __str__(self):
    p = "=" * 90 + "\n"
    p += f"Summary TaskGenerator Tasks: " + str(len(self._task_list)) + "\n"
    for j, t in enumerate(self._task_list):
      p += f"  {j:02d} Name: " + t.name + " " + "\n"
    p += "=" * 90
    return p

  def __len__(self):
    return self._total_tasks


def test():
  import sys
  import os

  sys.path.insert(0, os.getcwd())
  sys.path.append(os.path.join(os.getcwd() + "/src"))

  tg = TaskGenerator()
  print(tg)

  return True


if __name__ == "__main__":
  test()
