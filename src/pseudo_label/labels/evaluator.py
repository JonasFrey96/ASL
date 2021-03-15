import pandas as pd
from pytorch_lightning import metrics as pl_metrics
import torch
try:
  from generator import PseudoLabelGenerator
except:
  from .generator import PseudoLabelGenerator
 
__all__ = ['PseudoLabelEvaluator']

class PseudoLabelEvaluator():
  def __init__(self, settings):
    """
    settings: list of dicts with "name", "cfg" params for PLG
    """
    self.settings = settings
    self.df = pd.DataFrame( columns=['name', 'idx', 'acc_raw', 'acc_pseudo'] )
    self.acc = pl_metrics.classification.Accuracy()
    self.visu = None

  def evaluate_all_settings(self):
    for j, s in enumerate( self.settings):
      print(f"Evaluate Setting {j}/{len(self.settings)-1}: \n", s)
      plg = PseudoLabelGenerator(visu=self.visu, visu_active=False, **s['cfg'])
      
      for i in range(0,len(plg)):
        print(i, '/', len(plg))  
        dept, label_pseudo, label_pred_raw = plg.calculate_label(i)
        label_gt = plg.get_gt_label(i)
        m  =  label_gt > -1
        acc_pseudo = float( self.acc(torch.from_numpy( label_pseudo[m]), torch.from_numpy( label_gt[m])) )
        acc_raw = float( self.acc(torch.from_numpy( label_pred_raw[m]), torch.from_numpy( label_gt[m])) )
        self.df = self.df.append( {'name': s['name'], 'idx': i, 'acc_raw': acc_raw, 'acc_pseudo': acc_pseudo  },
              ignore_index=True)
  def print_results(self):
    return self.df

  def evaluate_df(self):
    return self.df.groupby('name').mean()

def test():
  settings = [
    {
    'name': "Super",
    'cfg': {  "refine_superpixel": True,
              "window_size": 3}
    },
    {
    'name': "Not Super",
    'cfg': {  "refine_superpixel": False,
              "window_size": 3}
    }
  ]    
  ple = PseudoLabelEvaluator(settings)
  ple.evaluate_all_settings()
  print( ple.evaluate_df() )

if __name__ == '__main__':
  test()