from pytorch_lightning.loggers.neptune import NeptuneLogger
import os
logger = NeptuneLogger(
          api_key=os.environ["NEPTUNE_API_TOKEN"],
          project_name="jonasfrey96/asl",
          experiment_id='ASL-400',
          close_after_fit = False,
        )
print(logger.experiment.id)

logger.experiment

print('Done')

print('Done')

print('Done')

print('Done')

print('Done')

print('Done')

import time
time.sleep(1)



