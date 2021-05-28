from .task_generator_scannet import TaskGeneratorScannet
from .task_generator_coco import TaskGeneratorCoco 

task_generators = {
    'coco': TaskGeneratorCoco ,
    'scannet': TaskGeneratorScannet
}

__all__ = ['get_task_generator']

def get_task_generator(name, **kwargs):
    return task_generators[name.lower()](**kwargs)