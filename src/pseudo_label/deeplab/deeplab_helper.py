import torch
import numpy as np
from torchvision import transforms as tf

import torch
import torch.nn.functional as F
import pickle

__all__ = ['DeeplabHelper']


class DeeplabHelper():
    def __init__(self, device,nc=40):
            self.model = torch.hub.load("kazuto1011/deeplab-pytorch", "deeplabv2_resnet101", pretrained='cocostuff164k', n_classes=182)
            self.model.eval()
            self.model.to(device)
            self.tra = tf.Resize((266,513))
            self.device = device

            with open('/home/jonfrey/ASL/cfg/dataset/mappings/coco_nyu.pkl', 'rb') as handle:
                mappings = pickle.load(handle)
            ls = [mappings['coco_id_nyu_id'][k] for k in mappings['coco_id_nyu_id'].keys() ]
            self.map_coco_nyu = torch.tensor(ls) 
            self._nc = nc

    def get_label(self, img):
        with torch.no_grad():
            # H,W,C 0-255 uint8 np
            
            H,W,C = img.shape
            img = torch.from_numpy( img.astype(np.float32) ).permute(2,0,1).to( self.device )[None]
            _, _, H, W = img.shape
            img = self.img_resize_normalize( img )
            # Image -> Probability map
            logits = self.model(img)
            logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
            probs = F.softmax(logits, dim=1)[0]
            
            
            label_coco = torch.argmax(probs, dim=0)
            sa = label_coco.shape
            label = label_coco.flatten()
            label_nyu = self.map_coco_nyu[label_coco.type(torch.int64)] # scannet to nyu40
            tra_up = tf.Resize((H,W))
            label_nyu = tra_up( label_nyu.reshape(sa)[None,:,:] )
            return label_nyu.cpu().numpy()[0]
    
    def get_label_prob(self, img):
        with torch.no_grad():
            # img: H,W,C 0-255 uint8 np
            # return: 41, H, W
            
            H,W,C = img.shape
            img = torch.from_numpy( img.astype(np.float32) ).permute(2,0,1).to( self.device )[None]
            _, _, H, W = img.shape
            img = self.img_resize_normalize( img )
            # Image -> Probability map
            logits = self.model(img)
            logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
            probs = F.softmax(logits, dim=1)[0]
            
            label_nyu = torch.zeros( self._nc+1,probs.shape[1],probs.shape[2], device=probs.device )
            for i in range( probs.shape[0] ) :
               label_nyu[ self.map_coco_nyu[i]+1 ] += probs[i]
            
            label_nyu = label_nyu / label_nyu.sum( dim=0 )
            # tra_up = tf.Resize((H,W))
            # label_nyu = tra_up( label_nyu.reshape(sa)[None,:,:] )
            return label_nyu.cpu().numpy()


    def img_resize_normalize(self, img):
        # Resize and normalize
        img = self.tra(img)
        img[:,0,:,:] -= 122.675
        img[:,1,:,:] -= 116.669
        img[:,2,:,:] -= 104.008
        return img

if __name__ == "__main__":
    import imageio
    import numpy as np

    def label_to_png2(label,path):
        max_classes = 40
        H,W,_ = label.shape 
        idxs = np.zeros( (3, H,W) ,dtype=np.uint8)
        values = np.zeros( (3, H,W) )
        label_c = label #.clone()
        max_val_10bit = 1023

        for i in range(3):
            idx = np.argmax( label_c, axis=2 )
            idxs[i] = idx.astype(np.int32)
            
            m = np.eye(max_classes)[idx] == 1
            values[i] = ( (label_c[m] *  max_val_10bit).reshape(H,W)).astype(np.int32)
            values[i][values[i] > max_val_10bit] = max_val_10bit
            label_c[m] = 0
        
        values = values.astype( np.int32)
        idxs = idxs.astype( np.int32)
        
        png = np.zeros( (H,W,4), dtype=np.int32 )
        for i in range(3):
            png[:,:,i] = values[i]
            png[:,:,i] = np.bitwise_or( png[:,:,i], idxs[i] << 10 )
        imageio.imwrite(path, png.astype(np.uint16),  format='PNG-FI', compression=9) 
    
        return True
    

    import sys
    import os
    sys.path.append("""/home/jonfrey/ASL/src""")
    os.chdir("/home/jonfrey/ASL")

    import imageio 
    i1 = imageio.imread( "/home/jonfrey/datasets/scannet/scans/scene0033_00/color/500.jpg" )
    yh = DeeplabHelper(device="cuda:0")
    label = yh.get_label_prob( i1 )
    
    from visu import Visualizer

    with open("/home/jonfrey/ASL/cfg/exp/create_newlabels/create_load_model.yml") as file:
        exp = yaml.load(file, Loader=yaml.FullLoader)
    scenes = exp['label_generation']['scenes']

    fsh = DeeplabHelper(device="cuda:0")
    
    paths = [str(s) for s in Path("/media/scratch1/jonfrey/scannet/scans/").rglob('*.jpg') 
                if  str(s).find("color") != -1 and 
                    str(s).split("/")[-3] in scenes] 
    paths = [p  for p in paths if int(p.split('/')[-1][:-4]) % 10 == 0]


    visu = Visualizer(os.getenv('HOME')+'/tmp', logger=None, epoch=0, store=True, num_classes=41)
    max_cores = 10
    with Pool(processes = max_cores) as pool:
        dir_out = "/media/scratch1/jonfrey/labels_generated"
        idtf = "labels_deeplab"
        for j, p in enumerate( paths ):
            scene = p.split("/")[-3]
            print( j,"/", len(paths) )
            i1 = imageio.imread(p)
            label = fsh.get_label_prob( i1 )

            idx = p.split('/')[-1][:-4]
            p_out = os.path.join( dir_out, idtf, scene, idtf, idx+'.png' )
            Path(p_out).parent.mkdir(exist_ok=True, parents= True)

            lab = np.swapaxes(np.swapaxes(label,0,2 ),0,1)
            _res = pool.apply_async(func= label_to_png2, args=(lab[:,:,1:] , p_out) )
            if j % max_cores == 0 and j != 0:
                _res.get()




    # import imageio 
    # i1 = imageio.imread( "/home/jonfrey/datasets/scannet/scans/scene0033_00/color/500.jpg" )
    
    
    # print(label.max(), label.shape)
    # from visu import Visualizer
    # visu = Visualizer(os.getenv('HOME')+'/tmp', logger=None, epoch=0, store=True, num_classes=41)
    # visu.plot_segmentation(seg=np.argmax(label, axis=0),jupyter=False, method='right')
    # visu.plot_image(img=i1,jupyter=False, method='left', tag="plot_segmentation")

