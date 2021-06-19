from .task_generator_scannet import TaskGeneratorScannet
from .task_generator_coco2014 import TaskGeneratorCoco2014
from .task_generator_cocostuff import TaskGeneratorCocoStuff

task_generators = {
    'coco2014': TaskGeneratorCoco2014,
    'scannet': TaskGeneratorScannet,
    'cocostuff': TaskGeneratorCocoStuff,
}

__all__ = ['get_task_generator']

def get_task_generator(name, **kwargs):
    return task_generators[name.lower()](**kwargs)