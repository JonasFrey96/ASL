from .task_generator_scannet import TaskGeneratorScannet

task_generators = {
    "scannet": TaskGeneratorScannet,
}

__all__ = ["get_task_generator"]


def get_task_generator(name, **kwargs):
    return task_generators[name.lower()](**kwargs)
