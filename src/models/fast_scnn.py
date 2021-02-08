###########################################################################
# Created by: Tramac
# Date: 2019-03-25
# Copyright (c) 2017
# https://github.com/Tramac/Fast-SCNN-pytorch 
# modified by Jonas Frey
# Licensed under Apache 2.0
###########################################################################

"""Fast Segmentation Convolutional Neural Network"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['FastSCNN', 'get_fast_scnn']

def inject(x,injection_features, injection_mask):
  if (x.shape[1] == injection_features.shape[1] and 
      x.shape[2] == injection_features.shape[2] and
      x.shape[3] == injection_features.shape[3] ):
    s = x.shape
    x = x + injection_features * injection_mask[:,:,None,None].repeat(1, s[1], s[2], s[3])
  return x

class FastSCNN(nn.Module):
  def __init__(self, num_classes, aux=False, **kwargs):
    super().__init__()
    self.aux = aux
    
    
    learning_to_downsample = LearningToDownsample(32, 48, 64)
    global_feature_extractor = GlobalFeatureExtractor(64, [64, 96, 128], 128, 6, [3, 3, 3])
    feature_fusion = FeatureFusionModule(64, 128, 128)
    classifier = Classifer(128, num_classes)
    
    self.extract = kwargs.get('extraction',{}).get('active', False)
    self.extract_layer = kwargs.get('extraction',{}).get('layer', 'extractor')
    
    self._md = nn.ModuleDict( 
      { 'learn_to_down': learning_to_downsample,
        'extractor': global_feature_extractor,
        'fusion': feature_fusion,
        'classifier': classifier  }
    )
    
    if self.aux:
      self.auxlayer = nn.Sequential(
        nn.Conv2d(64, 32, 3, padding=1, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(True),
        nn.Dropout(0.1),
        nn.Conv2d(32, num_classes, 1)
      )

  def forward(self, x):
    # generic extraction is implemented in a bad way. 
    extraction = None
    size = x.size()[2:] # 384,384,3 = 442368
    
    higher_res_features = self._md['learn_to_down'](x) # BS,64,48,48 = 147456
    if self.extract:
      if self.extract_layer == 'learn_to_down':
        extraction = higher_res_features.clone().detach()
      
    x = self._md['extractor'](higher_res_features) # BS,128,12,12 = 18432 Compression factor of 24
    if self.extract:
      if self.extract_layer == 'extractor':
        extraction = x.clone().detach()
    
    x = self._md['fusion'](higher_res_features, x) # BS,128,48,48 = 294912
    if self.extract:
      if self.extract_layer == 'fusion':
        extraction = x.clone().detach()
        
    x = self._md['classifier'](x) # BS,40,48,48
    outputs = []
    x = F.interpolate(x, size, mode='bilinear', align_corners=True)
    outputs.append(x)
    outputs.append(extraction) 
    
    if self.aux:
      auxout = self.auxlayer(higher_res_features)
      auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
      outputs.append(auxout)

    return tuple(outputs)
  
  def injection_forward(self, x, injection_features, injection_mask):
    size = x.size()[2:] # 384,384,3 = 442368
    x = inject(x, injection_features, injection_mask)
    
    # input replay
    higher_res_features = self._md['learn_to_down'](x) # BS,64,48,48 = 147456
    higher_res_features = inject(higher_res_features, injection_features, injection_mask)
    x = self._md['extractor'](higher_res_features) # BS,128,12,12 = 18432 Compression factor of 24
    x = inject(x, injection_features, injection_mask)
    x = self._md['fusion'](higher_res_features, x) # BS,128,48,48 = 294912   
    x = inject(x, injection_features, injection_mask)
    x = self._md['classifier'](x) # BS,40,48,48
    outputs = []
    x = F.interpolate(x, size, mode='bilinear', align_corners=True)
    outputs.append(x)
    
    if self.aux:
      auxout = self.auxlayer(higher_res_features)
      auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
      outputs.append(auxout)
    
    return tuple(outputs)
  

  def freeze_module(self, mask=[False,False,False,False], layer=None ):
    if layer is not None:
      mask = []
      k = list( self._md.keys()) 
      for i in range(10):
        if k[i] == layer:
          mask.append(True)
          break
        mask.append(True)
      if len(mask) < 4:
        mask = mask + [False]*(4-len(mask))  
    
    for m, mod in zip (mask, self._md.values()):
      if m:
        # mod.requires_grad = False
        for parameter in mod.parameters():
	        parameter.requires_grad = False
      else:
        for parameter in mod.parameters():
	        parameter.requires_grad = True
        # mod.requires_grad = True
  
class _ConvBNReLU(nn.Module):
  """Conv-BN-ReLU"""

  def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, **kwargs):
    super(_ConvBNReLU, self).__init__()
    self.conv = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(True)
    )

  def forward(self, x):
    return self.conv(x)

class _DSConv(nn.Module):
  """Depthwise Separable Convolutions"""

  def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
    super(_DSConv, self).__init__()
    self.conv = nn.Sequential(
      nn.Conv2d(dw_channels, dw_channels, 3, stride, 1, groups=dw_channels, bias=False),
      nn.BatchNorm2d(dw_channels),
      nn.ReLU(True),
      nn.Conv2d(dw_channels, out_channels, 1, bias=False),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(True)
    )

  def forward(self, x):
    return self.conv(x)

class _DWConv(nn.Module):
  def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
    super(_DWConv, self).__init__()
    self.conv = nn.Sequential(
      nn.Conv2d(dw_channels, out_channels, 3, stride, 1, groups=dw_channels, bias=False),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(True)
    )

  def forward(self, x):
    return self.conv(x)

class LinearBottleneck(nn.Module):
  """LinearBottleneck used in MobileNetV2"""

  def __init__(self, in_channels, out_channels, t=6, stride=2, **kwargs):
    super(LinearBottleneck, self).__init__()
    self.use_shortcut = stride == 1 and in_channels == out_channels
    self.block = nn.Sequential(
      # pw
      _ConvBNReLU(in_channels, in_channels * t, 1),
      # dw
      _DWConv(in_channels * t, in_channels * t, stride),
      # pw-linear
      nn.Conv2d(in_channels * t, out_channels, 1, bias=False),
      nn.BatchNorm2d(out_channels)
    )

  def forward(self, x):
    out = self.block(x)
    if self.use_shortcut:
      out = x + out
    return out


class PyramidPooling(nn.Module):
  """Pyramid pooling module"""

  def __init__(self, in_channels, out_channels, **kwargs):
    super(PyramidPooling, self).__init__()
    inter_channels = int(in_channels / 4)
    self.conv1 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
    self.conv2 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
    self.conv3 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
    self.conv4 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
    self.out = _ConvBNReLU(in_channels * 2, out_channels, 1)

  def pool(self, x, size):
    avgpool = nn.AdaptiveAvgPool2d(size)
    return avgpool(x)

  def upsample(self, x, size):
    return F.interpolate(x, size, mode='bilinear', align_corners=True)

  def forward(self, x):
    # BS, 128, 12, 12
    size = x.size()[2:]
    feat1 = self.upsample(self.conv1(self.pool(x, 1)), size)
    feat2 = self.upsample(self.conv2(self.pool(x, 2)), size)
    feat3 = self.upsample(self.conv3(self.pool(x, 3)), size)
    feat4 = self.upsample(self.conv4(self.pool(x, 6)), size)
    x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
    x = self.out(x)
    return x


class LearningToDownsample(nn.Module):
  """Learning to downsample module"""

  def __init__(self, dw_channels1=32, dw_channels2=48, out_channels=64, **kwargs):
    super(LearningToDownsample, self).__init__()
    self.conv = _ConvBNReLU(3, dw_channels1, 3, 2)
    self.dsconv1 = _DSConv(dw_channels1, dw_channels2, 2)
    self.dsconv2 = _DSConv(dw_channels2, out_channels, 2)

  def forward(self, x):
    # BS,3,384,384
    x = self.conv(x)
    x = self.dsconv1(x)
    x = self.dsconv2(x)
    return x


class GlobalFeatureExtractor(nn.Module):
  """Global feature extractor module"""

  def __init__(self, in_channels=64, block_channels=(64, 96, 128),
         out_channels=128, t=6, num_blocks=(3, 3, 3), **kwargs):
    super(GlobalFeatureExtractor, self).__init__()
    self.bottleneck1 = self._make_layer(LinearBottleneck, in_channels, block_channels[0], num_blocks[0], t, 2)
    self.bottleneck2 = self._make_layer(LinearBottleneck, block_channels[0], block_channels[1], num_blocks[1], t, 2)
    self.bottleneck3 = self._make_layer(LinearBottleneck, block_channels[1], block_channels[2], num_blocks[2], t, 1)
    self.ppm = PyramidPooling(block_channels[2], out_channels)

  def _make_layer(self, block, inplanes, planes, blocks, t=6, stride=1):
    layers = []
    layers.append(block(inplanes, planes, t, stride))
    for i in range(1, blocks):
      layers.append(block(planes, planes, t, 1))
    return nn.Sequential(*layers)

  def forward(self, x):
    # BS, 64, 24, 24
    x = self.bottleneck1(x)
    x = self.bottleneck2(x)
    x = self.bottleneck3(x)
    x = self.ppm(x)
    return x


class FeatureFusionModule(nn.Module):
  """Feature fusion module"""

  def __init__(self, highter_in_channels, lower_in_channels, out_channels, scale_factor=4, **kwargs):
    super(FeatureFusionModule, self).__init__()
    self.scale_factor = scale_factor
    self.dwconv = _DWConv(lower_in_channels, out_channels, 1)
    self.conv_lower_res = nn.Sequential(
      nn.Conv2d(out_channels, out_channels, 1),
      nn.BatchNorm2d(out_channels)
    )
    self.conv_higher_res = nn.Sequential(
      nn.Conv2d(highter_in_channels, out_channels, 1),
      nn.BatchNorm2d(out_channels)
    )
    self.relu = nn.ReLU(True)

  def forward(self, higher_res_feature, lower_res_feature):
    lower_res_feature = F.interpolate(lower_res_feature, scale_factor=4, mode='bilinear', align_corners=True)
    lower_res_feature = self.dwconv(lower_res_feature)
    lower_res_feature = self.conv_lower_res(lower_res_feature)

    higher_res_feature = self.conv_higher_res(higher_res_feature)
    out = higher_res_feature + lower_res_feature
    return self.relu(out)


class Classifer(nn.Module):
  """Classifer"""

  def __init__(self, dw_channels, num_classes, stride=1, **kwargs):
    super(Classifer, self).__init__()
    self.dsconv1 = _DSConv(dw_channels, dw_channels, stride)
    self.dsconv2 = _DSConv(dw_channels, dw_channels, stride)
    self.conv = nn.Sequential(
      nn.Dropout(0.1),
      nn.Conv2d(dw_channels, num_classes, 1)
    )

  def forward(self, x):
    # BS, 40, 48, 48
    x = self.dsconv1(x)
    x = self.dsconv2(x)
    x = self.conv(x)
    return x


def get_fast_scnn(dataset='citys', pretrained=False, root='./weights', map_cpu=False, **kwargs):
  acronyms = {
    'pascal_voc': 'voc',
    'pascal_aug': 'voc',
    'ade20k': 'ade',
    'coco': 'coco',
    'citys': 'citys',
  }
  from data_loader import datasets
  model = FastSCNN(datasets[dataset].NUM_CLASS, **kwargs)
  if pretrained:
    if(map_cpu):
      model.load_state_dict(torch.load(os.path.join(root, 'fast_scnn_%s.pth' % acronyms[dataset]), map_location='cpu'))
    else:
      model.load_state_dict(torch.load(os.path.join(root, 'fast_scnn_%s.pth' % acronyms[dataset])))
  return model


def test_input_size():
  # pytest -q -s src/models/fast_scnn.py
  
  model = FastSCNN( 41, aux=False)
  input_sizes = [(382,382),(480,640),(764,764)]
  C = 3
  BS = 8
  for s1,s2 in input_sizes:
    data = torch.rand( (BS,C,s1,s2), dtype=torch.float32)
    print( 'Input', data.shape)
    res = model(data)
    print( 'Output', res[0].shape )
    
if __name__ == '__main__':
  test_input_size()
  # img = torch.randn(2, 3, 256, 512)
  # model = get_fast_scnn('citys')
  # outputs = model(img)