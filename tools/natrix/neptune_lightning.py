from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning import Trainer
import neptune
import os

def _create_or_get_experiment2(self):
  proxies = {
  'http': 'http://proxy.ethz.ch:3128',
  'https': 'http://proxy.ethz.ch:3128',
  }
  if self.offline_mode:
      project = neptune.Session(backend=neptune.OfflineBackend()).get_project('dry-run/project')
  else:
      #project_qualified_name='jonasfrey96/ASL', api_token=os.environ["NEPTUNE_API_TOKEN"], proxies=proxies
      session = neptune.init(project_qualified_name='jonasfrey96/ASL', api_token=self.api_key,proxies=proxies) # add your credential
      print(type(session))
      session = neptune.Session(api_token=self.api_key,proxies=proxies)
      project = session.get_project(self.project_name)

  if self.experiment_id is None:
      e = project.create_experiment(name=self.experiment_name, **self._kwargs)
      self.experiment_id = e.id
  else:
      e = project.get_experiments(id=self.experiment_id)[0]
      self.experiment_name = e.get_system_properties()['name']
      self.params = e.get_parameters()
      self.properties = e.get_properties()
      self.tags = e.get_tags()
  return e
NeptuneLogger._create_or_get_experiment = _create_or_get_experiment2 # Super bad !!!

logger = NeptuneLogger(
        api_key=os.environ["NEPTUNE_API_TOKEN"],
        project_name="jonasfrey96/asl",
        experiment_name= 'Best_Project',
        params={'test':42}, 
        tags=['thanks'],
        close_after_fit = False,
        offline_mode = False
    )
trainer = Trainer(logger=logger) 