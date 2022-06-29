import copy
from ucdr.task import TaskGenerator, Task
import numpy as np

__all__ = ["TaskGeneratorScannet"]

scannet_template_dict = {
    "name": "scannet",
    "mode": "train",
    "output_size": [320, 640],
    "scenes": [],
    "data_augmentation": True,
    "label_setting": "default",
    "confidence_aux": 0,
}


class TaskGeneratorScannet(TaskGenerator):
    def __init__(self, mode, cfg, *args, **kwargs):
        # SET ALL TEMPLATES CORRECT
        super(TaskGeneratorScannet, self).__init__(cfg)

        for k in cfg["copy_to_template"].keys():
            scannet_template_dict[k] = cfg["copy_to_template"][k]

        mode_cfg = cfg.get(mode, {})
        if mode == "scannet_scenes":
            self._scannet_scenes(**mode_cfg)
        elif mode == "scannet_pretrain":
            self._scannet_pretrain(**mode_cfg)
        elif mode == "scannet_auxilary_labels":
            self._scannet_auxilary_labels(**mode_cfg)
        elif mode == "scannet_25k":
            self._scannet_25k(**mode_cfg)
        elif mode == "scannet_25k_individual":
            self._scannet_25k_individual(**mode_cfg)
        elif mode == "scannet_25k_alternating":
            self._scannet_25k_alternating(**mode_cfg)
        elif mode == "scannet_25k_reference":
            self._scannet_25k_reference(**mode_cfg)
        else:
            raise AssertionError("TaskGeneratorScannet: Undefined Mode")

        self.init_end_routine(cfg)

    def _scannet_auxilary_labels(self, label_setting="default"):
        train = copy.deepcopy(scannet_template_dict)
        val = copy.deepcopy(scannet_template_dict)
        train["mode"] = "train"
        val["mode"] = "val"

        # Define the first pretrain task
        train["scenes"] = [f"scene{s:04d}" for s in range(10, 60)]
        val["scenes"] = train["scenes"]
        i = 0
        t = Task(
            name=f"Train_{i}",
            dataset_train_cfg=copy.deepcopy(train),
            dataset_val_cfg=copy.deepcopy(val),
        )
        self._task_list.append(t)

        start_scene_train = 0
        scenes_per_task = 1
        for i in range(1):
            # GENERATE TRAIN TASK
            train["scenes"] = [f"scene{s:04d}" for s in range(start_scene_train, start_scene_train + scenes_per_task)]
            train["label_setting"] = label_setting

            val["scenes"] = train["scenes"]
            val["label_setting"] = label_setting

            t = Task(
                name=f"Train_{i+1}",
                dataset_train_cfg=copy.deepcopy(train),
                dataset_val_cfg=copy.deepcopy(val),
            )
            self._task_list.append(t)
            start_scene_train += scenes_per_task

    def _scannet_25k(self):
        train = copy.deepcopy(scannet_template_dict)
        val = copy.deepcopy(scannet_template_dict)
        train["mode"] = "train_25k"
        val["mode"] = "val_25k"
        val["data_augmentation"] = False
        t = Task(
            name=f"Train_25k",
            dataset_train_cfg=copy.deepcopy(train),
            dataset_val_cfg=copy.deepcopy(val),
        )
        self._task_list.append(t)

    def _scannet_25k_alternating(self, number_of_tasks, scenes_per_task, label_setting, confidence_aux=0):
        self._scannet_25k()
        self._scannet_scenes(number_of_tasks, scenes_per_task, label_setting, confidence_aux)

        idx = 2
        while idx < len(self._task_list):
            self._task_list.insert(idx, copy.deepcopy(self._task_list[0]))
            idx += 2

    def _scannet_25k_individual(
        self,
        number_of_tasks,
        scenes_per_task,
        label_setting,
        confidence_aux=0,
        start_scene=0,
    ):
        self._scannet_25k()
        self._scannet_scenes(
            number_of_tasks,
            scenes_per_task,
            label_setting,
            confidence_aux,
            start_scene=start_scene,
        )

    def _scannet_25k_reference(self, number_of_tasks, scenes_per_task, label_setting, confidence_aux=0):
        self._scannet_25k()
        self._scannet_scenes(number_of_tasks, scenes_per_task, label_setting, confidence_aux)
        sce = [t.dataset_train_cfg["scenes"] for t in self._task_list[1:]]
        sce = np.array([t.dataset_train_cfg["scenes"] for t in self._task_list[1:]]).flatten().tolist()
        self._task_list[1].dataset_train_cfg["scenes"] = sce
        self._task_list[1].dataset_val_cfg["scenes"] = sce

    def _scannet_pretrain(self):
        train = copy.deepcopy(scannet_template_dict)
        val = copy.deepcopy(scannet_template_dict)
        train["mode"] = "train"
        val["mode"] = "val"
        val["data_augmentation"] = False
        train["scenes"] = [f"scene{s:04d}" for s in range(10, 60)]
        val["scenes"] = train["scenes"]
        t = Task(
            name=f"Train_0",
            dataset_train_cfg=copy.deepcopy(train),
            dataset_val_cfg=copy.deepcopy(val),
        )
        self._task_list.append(t)

    def _scannet_scenes(
        self,
        number_of_tasks,
        scenes_per_task,
        label_setting="default",
        confidence_aux=0,
        start_scene=0,
    ):
        train = copy.deepcopy(scannet_template_dict)
        val = copy.deepcopy(scannet_template_dict)
        train["mode"] = "train"
        val["mode"] = "val"
        train["label_setting"] = label_setting
        train["confidence_aux"] = confidence_aux

        val["label_setting"] = label_setting  # "default"
        val["data_augmentation"] = False

        start_scene_train = start_scene
        for i in range(number_of_tasks):
            # GENERATE TRAIN TASK
            train["scenes"] = [f"scene{s:04d}" for s in range(start_scene_train, start_scene_train + scenes_per_task)]
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

    from ucdr.utils import load_yaml

    exp = load_yaml(os.path.join(os.getcwd() + "/cfg/test/test.yaml"))

    tg = TaskGeneratorScannet(mode=exp["task_generator"]["mode"], cfg=exp["task_generator"]["cfg"])
    print(tg)
    tg = TaskGeneratorScannet(mode="scannet_25k", cfg=exp["task_generator"]["cfg"])
    print(tg)
    tg = TaskGeneratorScannet(mode="scannet_25k_alternating", cfg=exp["task_generator"]["cfg"])

    print(tg)
    return True


if __name__ == "__main__":
    test()
