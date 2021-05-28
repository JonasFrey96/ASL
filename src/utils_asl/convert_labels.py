import imageio
import numpy as np

__all__ = ['label_to_png', 'png_to_label']

def label_to_png(label, path, max_classes = 40):
  assert len(label.shape) == 3
  assert label.shape[2] == max_classes
  H,W,_ = label.shape 
  idxs = np.zeros( (3, H,W) ,dtype=np.uint8 )
  values = np.zeros( (3, H,W) )
  label_c = np.copy( label )
  max_val_10bit = 1023
  
  for i in range(3):
    idx = np.argmax( label_c, axis=2 )
    idxs[i] = np.uint8(idx)
    
    m = np.eye(max_classes)[idx] == 1
    values[i] = np.uint16( (label_c[m] *  max_val_10bit)).reshape(H,W)
    values[i][values[i] > max_val_10bit] = max_val_10bit
    label_c[m] = 0
    
  png = np.zeros( (H,W,4), dtype=np.uint16)
  
  for i in range(3):
    png[:,:,i] = values[i]
    png[:,:,i] = np.bitwise_or( png[:,:,i], np.left_shift( idxs[i] ,10, dtype=np.uint16))
    
  imageio.imwrite(path, png,  format='PNG-FI') 
  

def png_to_label(path, max_classes = 40):
  png = imageio.imread(path,  format='PNG-FI')
  H,W,_ = png.shape
  label = np.zeros( ( H,W,max_classes) )
  
  iu16 = np.iinfo( np.uint16)
  
  mask = np.full( (H,W), iu16.max, dtype=np.uint16)
  mask_low = np.right_shift( mask, 6, dtype=np.uint16) 

  for i in range(3):
    
    prob = np.bitwise_and( png[:,:,i], mask_low) / 1023
    cls = np.right_shift(png[:,:,i], 10, dtype=np.uint16)

    m = np.eye(max_classes)[cls] == 1
    label[m] = prob.reshape(-1)
    
  return label
  
def test():
  # Fully working
  H,W = 480,640 
  label = np.random.random( (H,W,40))
  label_to_png(label, path='test.png')
  lo = png_to_label(path= 'test.png', max_classes = 40)
  print( np.mean( np.abs(  lo[lo != 0] - label[lo != 0] )  ) )
  
  
if __name__ == '__main__':
  test()