from .cityscapes import Cityscapes
from .nyu_v2 import NYUv2
from .ml_hypersim import MLHypersim
from .coco import COCo

datasets = {
    'cityscapes': Cityscapes,
    'nyuv2': NYUv2,
    'mlhypersim': MLHypersim,
    'coco': COCo     
}

def get_dataset(name, env, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](root=env[name], **kwargs)

    
def test():
    # pytest -q -s src/datasets/datasets_collector.py
    import yaml
    import os 
    from pathlib import Path
    import numpy as np
    import imageio
    from visu import Visualizer
    import time
    
    vis = Visualizer( 'src/datasets/test_results/', writer=None, epoch=0, store=True, num_classes=41)
    
    home = str()
    with open(Path.joinpath( Path.home(),"ASL/cfg/env/env.yml")) as file:  
        env = yaml.load(file, Loader=yaml.FullLoader) 
    configs = [
        {'name':'coco',
         'squeeze_80_labels_to_40': True},
        {'name':'mlhypersim'},
        {'name':'nyuv2'}
        # {'name':'cityscapes'}
    ]
    for config in configs:
        print(config)
        dataset = get_dataset(**config, env=env)
        n = config['name']
        t_total = 0
        plot = False
        for i in range(0,10):
            st = time.time() 
            img, label = dataset[i]    # C, H, W
            t_total += time.time()-st
            if plot:
                vis.plot_segmentation(seg = label, tag = f'{n}_label', epoch = i)
                
                label = np.uint8( label.numpy() * (255/float(label.max())))[:,:]
                img = np.uint8( img.permute(1,2,0).numpy()*255 ) # H W C
                
                imageio.imwrite(f'src/datasets/test_results/{i}_{n}_img.png', img)
            
            print( f'Name: {n}, Iter: {i}')
        print(f'Total loading time for {n}: {t_total}s')