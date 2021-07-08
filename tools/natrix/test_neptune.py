# export your NEPTUNE_API_TOKEN
# before starting python script:
# module load eth_proxy
import neptune
import os
import time

proxies = {
  "http": "http://proxy.ethz.ch:3128",
  "https": "http://proxy.ethz.ch:3128",
}
neptune.init(
  project_qualified_name="jonasfrey96/ASL",
  api_token=os.environ["NEPTUNE_API_TOKEN"],
  proxies=proxies,
)
neptune.create_experiment()
