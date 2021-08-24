__all__ = ["_create_or_get_experiment2"]
import neptune


def _create_or_get_experiment2(self):
  """
  Super bad !!! Dont do this
  """
  proxies = {
    "http": "http://proxy.ethz.ch:3128",
    "https": "http://proxy.ethz.ch:3128",
  }
  if self.offline_mode:
    project = neptune.Session(backend=neptune.OfflineBackend()).get_project(
      "dry-run/project"
    )
  else:
    session = neptune.init(
      project_qualified_name="ASL/MT-JonasFrey", api_token=self.api_key, proxies=proxies
    )  # add your credential
    print(type(session))
    session = neptune.Session(api_token=self.api_key, proxies=proxies)
    project = session.get_project(self.project_name)

  if self.experiment_id is None:
    e = project.create_experiment(name=self.experiment_name, **self._kwargs)
    self.experiment_id = e.id
  else:
    e = project.get_experiments(id=self.experiment_id)[0]
    self.experiment_name = e.get_system_properties()["name"]
    self.params = e.get_parameters()
    self.properties = e.get_properties()
    self.tags = e.get_tags()
  return e
